from typing import Any, Dict, Tuple, Union

import torch
import numpy as np

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.torch.algos.algo import TorchRLAlgo

class TorchRecurrentAlgo(MethodWrapper):
    """
    A wrapper to burn in hidden states when training a recurrent algorithm.
    """
    def __init__(
            self,
            algo: TorchRLAlgo,
            burn_in_length: int = 0,
            n_steps: int = 1
        ):
        """
        Creates the recurrent algorithm wrapper on the underlying algorithm.

        Args:
            algo: The algorithm to wrap.
            burn_in_length: The number of states to burn in the hidden state.
            n_steps: The number of steps between the state and next states
                (needed to get the next hidden states properly)
        """
        super().__init__(algo)

        self.burn_in_length = burn_in_length
        self.n_steps = n_steps

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
        rollouts = {
            key: value.to(self.device) for key, value in rollouts.items()
        }

        # Switch from (batch size, 2, num layers, hidden size) to
        # (2, num layers, batch size, hidden size)
        rollouts["hidden_state"] = (
            rollouts["hidden_state"].permute(1, 2, 0, 3).contiguous()
        )

        states = rollouts["state"]
        actions = rollouts["action"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]
        hidden_states = rollouts["hidden_state"]

        burn_in_states = states
        new_hiddens = hidden_states

        if self.burn_in_length > 0:    
            with torch.no_grad():
                burn_in_states = states[:, :self.burn_in_length].contiguous()

                *_, new_hiddens = self.forward(
                    burn_in_states, hidden_states
                )


        rollouts["hidden_state"] = [nh for nh in new_hiddens]

        rollouts["state"] = states[:, self.burn_in_length:].contiguous()
        rollouts["action"] = actions[:, self.burn_in_length:].contiguous()
        rollouts["reward"] = rewards[:, self.burn_in_length:].contiguous()
        rollouts["next_state"] = next_states[
            :, self.burn_in_length:
        ].contiguous()
        rollouts["terminal"] = terminals[:, self.burn_in_length:].contiguous()

        with torch.no_grad():
            first_burned_in = rollouts["state"][:, :1].contiguous()
            *_, next_hiddens = self.forward(
                first_burned_in, new_hiddens
            )

        rollouts["next_hidden_state"] = [nh for nh in next_hiddens]

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
