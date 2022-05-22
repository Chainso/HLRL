from typing import Any, Dict, Union

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from hlrl.torch.utils.rnn import unpad_sequence
from hlrl.core.common.wrappers import MethodWrapper
from hlrl.torch.algos.algo import TorchRLAlgo

class TorchRecurrentAlgo(MethodWrapper):
    """
    A wrapper to burn in hidden states when training a recurrent algorithm.
    """
    def __init__(
            self,
            algo: TorchRLAlgo,
            burn_in_length: int = 0
        ):
        """
        Creates the recurrent algorithm wrapper on the underlying algorithm.

        Args:
            algo: The algorithm to wrap.
            burn_in_length: The number of states to burn in the hidden state.
        """
        super().__init__(algo)

        self.burn_in_length = burn_in_length

    def burn_in_hidden_states(self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        """
        Burns in the hidden states of the rollouts.

        Args:
            rollouts: The training batch to burn in hidden states for.

        Returns:
            The training batch with the hidden states burned in.
        """
        # Switch from (batch size, 2, num layers, hidden size) to
        # (2, num layers, batch size, hidden size)
        rollouts["hidden_state"] = (
            rollouts["hidden_state"].permute(1, 2, 0, 3).contiguous()
        )

        states = rollouts["state"]
        hidden_states = rollouts["hidden_state"]
        n_steps = rollouts["n_steps"]

        rollouts["sequence_length"] = rollouts["sequence_length"].to("cpu")
        rollouts["sequence_start"] = rollouts["sequence_start"].to("cpu")

        sequence_lengths = rollouts["sequence_length"]
        sequence_starts = rollouts["sequence_start"]

        # Need to unpad, split between burn in and sequence, repad then burn in
        # states
        if self.burn_in_length > 0:
            with torch.no_grad():
                states = unpad_sequence(
                    states, sequence_lengths, batch_first=True
                )

                burn_in = sequence_starts > 0

                burn_in_states = [
                    states[i][:sequence_starts[i]]
                    for i in range(len(states))
                    if burn_in[i]
                ]
                seq_states = [
                    states[i][sequence_starts[i]:]
                    for i in range(len(states))
                ]

                burn_seq_starts = sequence_starts[burn_in]
                burn_hidden = hidden_states[:, :, burn_in]

                burn_in_states = pad_sequence(burn_in_states, batch_first=True)
                seq_states = pad_sequence(seq_states, batch_first=True)

                *_, new_hiddens = self.forward(
                    burn_in_states, burn_hidden, lengths=burn_seq_starts
                )

                burned_in_hidden = torch.zeros_like(hidden_states)
                burned_in_hidden[:, :, burn_in] = torch.stack(
                    new_hiddens, dim=0
                )
                burned_in_hidden[:, :, ~burn_in] = hidden_states[
                    :, :, ~burn_in
                ]

                rollouts["state"] = seq_states
                rollouts["hidden_state"] = [nh for nh in burned_in_hidden]

        with torch.no_grad():
            n_step_next = [
                rollouts["state"][i][:n_steps[i, 0, 0]]
                for i in range(len(states))
            ]
            n_step_next = pad_sequence(n_step_next, batch_first=True)

            *_, next_hiddens = self.forward(
                n_step_next, rollouts["hidden_state"],
                lengths=n_steps[:, 0, 0].to("cpu")
            )

        rollouts["next_hidden_state"] = [nh for nh in next_hiddens]
        rollouts["sequence_length"] = sequence_lengths - sequence_starts

        seq_range = torch.arange(rollouts["state"].shape[1]).unsqueeze(0)
        rollouts["sequence_mask"] = (
            seq_range < sequence_lengths.unsqueeze(-1)
        ).unsqueeze(-1).to(rollouts["state"].device)

        return rollouts

    def process_batch(
            self,
            rollouts: Dict[str, Union[torch.Tensor, np.ndarray]],
            *args: Any,
            **kwargs: Any
        ) -> Dict[str, torch.Tensor]:
        """
        Processes a batch to make it suitable for training.

        Args:
            rollouts: The training batch to process.
            args: Any positional arguments for the wrapped algorithm to process
                the batch.
            kwargs: Any keyword arguments for the wrapped algorithm to process
                the batch.

        Returns:
            The processed training batch.
        """
        rollouts, *rest = self.om.process_batch(rollouts, *args, **kwargs)
        rollouts = self.burn_in_hidden_states(rollouts)

        return rollouts, *rest
