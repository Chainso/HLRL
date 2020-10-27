import torch
import queue

from collections import deque
from typing import Any, Dict, Tuple

from hlrl.core.common.wrappers import MethodWrapper
from .agent import TorchRLAgent

class SequenceInputAgent(TorchRLAgent):
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
        return super().make_tensor([data])

    def transform_action(self, action: torch.Tensor):
        """
        Remove the sequence axis for the environment.

        Args:
            action: The action to take in the environment.
        """
        transed_action = super().transform_action(action)
        transed_action = transed_action.squeeze(0)

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
        self.ready_experiences = {}
        self.num_experiences = 0

    def reset(self):
        """
        Clears the ready experience buffer in addition to the regular reset.
        """
        self.ready_experiences = {}
        self.num_experiences = 0

        self.om.reset()

    def get_buffer_experience(self,
                              experiences: Tuple[Dict[str, Any], ...],
                              decay: float) -> Any:
        """
        Perpares the experience to add to the buffer.

        Args:
            experiences: The experiences containing rewards.
            decay: The decay constant.

        Returns:
            The oldest stored experience.
        """
        experience = super().get_buffer_experience(experiences, decay)

        next_q_val = experience.pop("next_q_val")
        target_q_val = experience["reward"] + decay * next_q_val

        # Update experience with target q value
        experience["target_q_val"] = target_q_val

        return experience

    def add_to_replay_buffer(self, experience_queue, experiences, decay):
        """
        Adds the experience to the replay buffer.
        """
        experience = self.get_buffer_experience(experiences, decay)
        
        for key in experience:
            if key not in self.ready_experiences:
                self.ready_experiences[key] = []

            self.ready_experiences[key].append(experience[key])

        self.num_experiences += 1

        if self.num_experiences == self.sequence_length:
            # Concatenate experiences first
            experiences_to_send = {}
            for key in self.ready_experiences:
                if key == "hidden_state" or not key.endswith("hidden_state"):
                    if key == "hidden_state":
                        # Only need the first hidden state
                        experiences_to_send[key] = (
                            self.ready_experiences[key][0].permute(2, 0, 1, 3)
                        )
                    else:
                        # Concatenate to sequence dimension
                        experiences_to_send[key] = torch.cat(
                            self.ready_experiences[key], dim=1
                        )

            try:
                experience_queue.put_nowait(experiences_to_send)
            except queue.Full:
                pass

            keep_start = len(self.ready_experiences) - self.overlap
            self.num_experiences = self.overlap

            for key in self.ready_experiences:
                self.ready_experiences[key] = (
                    self.ready_experiences[key][keep_start:]
                )

