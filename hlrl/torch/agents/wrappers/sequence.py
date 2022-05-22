from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn.utils import rnn

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.torch.agents.wrappers import TorchRLAgent

class SequenceInputAgent(MethodWrapper):
    """
    An agent that provides sequences of input to the model (of length 1).
    """
    def make_tensor(self, data: Any, *args, **kwargs):
        """
        Creates a ftensor of the data of batch size 1, and sequence length of 1.

        Args:
            data: The data to transform into a tensor.
            args: Underlying arguments for make_tensor.
            kwargs: Underlying keyword arguments for make_tensor.
        """
        return self.om.make_tensor([data], *args, **kwargs)

    def transform_action(self, action: torch.Tensor):
        """
        Remove the sequence axis for the environment.

        Args:
            action: The action to take in the environment.
        """
        transed_action = action.squeeze(1)
        transed_action = self.om.transform_action(transed_action)

        return transed_action


class ExperienceSequenceAgent(MethodWrapper):
    """
    An agent that inputs a sequence of experiences to the replay buffer instead
    of one at a time.
    """
    def __init__(
            self,
            agent: TorchRLAgent,
            sequence_length: int,
            overlap: int = 0
        ):
        """
        Args:
            agent: The agent to wrap.
            sequence_length: The length of the sequences.
            overlap: Keeps the last n experiences from the previous batch.
        """
        super().__init__(agent)

        self.sequence_length = sequence_length
        self.overlap = overlap
        self.true_seq_length = sequence_length - overlap

        # Store it in dict to list format
        self.sequence_experiences = {}
        self.sequence_starts = None

    def reset(self):
        """
        Clears the ready experience buffer in addition to the regular reset.
        """
        self.sequence_experiences = {}
        self.sequence_starts = None

        self.om.reset()

    def prepare_experiences(
            self,
            experiences: Tuple[Dict[str, Any], ...],
        ) -> List[Dict[str, torch.Tensor]]:
        """
        Perpares the experiences to add to the buffer.

        Args:
            experiences: The experiences to add.

        Returns:
            The prepared experiences to add to the replay buffer.
        """
        experiences = self.om.prepare_experiences(experiences)
        prepared_sequences = []

        for experience in experiences:
            for key in experience:
                if key not in self.sequence_experiences:
                    # Create np array of list objects
                    self.sequence_experiences[key] = [
                        [] for _ in range(len(experience[key]))
                    ]

                    if self.sequence_starts is None:
                        self.sequence_starts = torch.zeros(
                            len(experience[key]), dtype=torch.long
                        )


                if key == "hidden_state":
                    for i in range(len(experience[key][0])):
                        self.sequence_experiences[key][i].append(
                            experience[key][:, i]
                        )
                else:
                    for i in range(len(experience[key])):
                        self.sequence_experiences[key][i].append(
                            experience[key][i]
                        )

            for i in range(len(self.sequence_starts)):
                terminal = (
                    experience["env_terminal"][i] or experience["terminal"][i]
                )
                seq_len = len(self.sequence_experiences["terminal"][i])

                if (terminal
                    or seq_len
                        - self.sequence_starts[i] == self.true_seq_length):
                    # Create a experience sequence up until the terminal or end
                    prepared_sequence = {}

                    prepared_sequence["sequence_start"] = self.sequence_starts[
                        i
                    ].clone().view(1)
                    prepared_sequence["sequence_length"] = torch.tensor([
                        len(self.sequence_experiences["terminal"][i])
                    ]).view(1)

                    for key in self.sequence_experiences:
                        if "hidden_state" in key:
                            # Only need the first hidden state
                            prepared_sequence[key] = (
                                self.sequence_experiences[key][i][0].unsqueeze(
                                    0
                                )
                            )
                        elif "state" == key:
                            # Stack to sequence dimension
                            seq_tens = torch.cat(
                                self.sequence_experiences[key][i]
                            )
                            padding = (
                                [0] * 2 * (len(seq_tens.shape) - 1)
                                + [0, self.sequence_length - seq_tens.shape[0]]
                            )

                            prepared_sequence[key] = nn.functional.pad(
                                seq_tens, padding
                            ).unsqueeze(0)
                        else:
                            # For all other tensors, only need the main sequence
                            # Not the burn in portion
                            seq_tens = torch.cat(
                                self.sequence_experiences[key][i][self.sequence_starts[i]:]
                            )
                            padding = (
                                [0] * 2 * (len(seq_tens.shape) - 1)
                                + [
                                    0,
                                    self.sequence_length - self.overlap
                                    - seq_tens.shape[0]
                                ]
                            )

                            prepared_sequence[key] = nn.functional.pad(
                                seq_tens, padding
                            ).unsqueeze(0)

                    removal_start = seq_len if terminal else -self.overlap

                    for key in self.sequence_experiences:
                        self.sequence_experiences[key][i] = (
                            self.sequence_experiences[key][i][removal_start:]
                        )

                    self.sequence_starts[i] = (
                        len(self.sequence_experiences["terminal"][i])
                    )

                    prepared_sequences.append(prepared_sequence)

        return prepared_sequences
