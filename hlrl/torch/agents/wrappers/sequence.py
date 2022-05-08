import torch
import numpy as np

from typing import Any, Dict, List, Tuple

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.torch.agents import TorchRLAgent

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
        ) -> Any:
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
                            len(experience[key]), device=experience[key].device
                        )

                for i in range(len(experience[key])):
                    self.sequence_experiences[key][i].append(experience[key][i])

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

                    for key in self.sequence_experiences:
                        if key == "hidden_state":
                            # Only need the first hidden state
                            prepared_sequence[key].append(
                                self.sequence_experiences[key][i][0].permute(
                                    2, 0, 1, 3
                                )
                            )
                        else:
                            # Stack to sequence dimension
                            prepared_sequence[key].append(torch.stack(
                                self.sequence_experiences[key][i], dim=1
                            ))

                    prepared_sequence["sequence_start"] = (
                        self.sequence_starts[i].unsqueeze(0)
                    )

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

    def add_to_buffer(
            self,
            ready_experiences: Dict[str, List[Any]],
            experiences: Tuple[Dict[str, Any], ...]
        ) -> None:
        """
        Prepares the oldest experiences from experiences and transfers it to
        ready experiences.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            experiences: The experiences containing rewards.
        """
        experiences = self.prepare_experiences(experiences)

        if self.num_experiences == self.sequence_length:
            # Concatenate experiences first
            for key in self.sequence_experiences:
                if key not in ready_experiences:
                    ready_experiences[key] = []

                if key == "hidden_state":
                    # Only need the first hidden state
                    ready_experiences[key].append(
                        self.sequence_experiences[key][0].permute(
                            2, 0, 1, 3
                        )
                    )
                else:
                    # Concatenate to sequence dimension
                    ready_experiences[key].append(torch.cat(
                        self.sequence_experiences[key], dim=1
                    ))

            # Remove the first sequence and keep the rest
            self.num_experiences -= self.overlap

            for key in self.sequence_experiences:
                self.sequence_experiences[key] = (
                    self.sequence_experiences[key][-self.overlap:]
                )
