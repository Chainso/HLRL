import torch
import queue

from collections import deque
from typing import Any, Dict, List, Tuple

from hlrl.core.common.wrappers import MethodWrapper
from .agent import TorchRLAgent

class SequenceInputAgent(MethodWrapper):
    """
    An agent that provides sequences of input to the model (of length 1).
    """
    def make_tensor(self, data: Any):
        """
        Creates a float tensor of the data of batch size 1, and sequence length
        of 1.

        Args:
            data: The data to transform into a tensor.
        """
        return self.om.make_tensor([data])

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
    def __init__(self, agent, sequence_length, overlap=0):
        """
        Args:
            agent (TorchRLAgent): The agent to wrap.
            sequence_length (int): The length of the sequences.
            overlap (int): Keeps the last n experiences from the previous
                batch.
        """
        super().__init__(agent)

        self.sequence_length = sequence_length
        self.overlap = overlap

        # Store it in list format
        self.sequence_experiences = {}
        self.num_experiences = 0

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Reduces the inputs used to serialize and recreate the experience
        sequence agent.

        Returns:
            A tuple of the class and input arguments.
        """
        return (type(self), (self.obj, self.sequence_length, self.overlap))

    def reset(self):
        """
        Clears the ready experience buffer in addition to the regular reset.
        """
        self.sequence_experiences = {}
        self.num_experiences = 0

        self.om.reset()

    def add_to_buffer(self,
                      ready_experiences: Dict[str, List[Any]],
                      experiences: Tuple[Dict[str, Any], ...],
                      decay: float) -> None:
        """
        Prepares the oldest experiences from experiences and transfers it to
        ready experiences.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            experiences: The experiences containing rewards.
            decay: The decay constant.
        """
        experience = self.get_buffer_experience(experiences, decay)
        
        for key in experience:
            if key not in self.sequence_experiences:
                self.sequence_experiences[key] = []

            self.sequence_experiences[key].append(experience[key])

        self.num_experiences += 1

        if self.num_experiences == self.sequence_length:
            # Concatenate experiences first
            for key in self.sequence_experiences:
                if key == "hidden_state" or not key.endswith("hidden_state"):
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
            self.num_experiences -= 1

            for key in self.sequence_experiences:
                self.sequence_experiences[key] = (
                    self.sequence_experiences[key][1:]
                )

