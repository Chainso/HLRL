import torch

from collections import deque

from hlrl.core.utils import MethodWrapper


class SequenceInputAgent(MethodWrapper):
    """
    An agent that provides sequences of input to the model (of length 1).
    """
    def __init__(self, agent):
        super().__init__(agent)

    def make_tensor(self, data):
        """
        Creates a float tensor of the data of batch size 1.
        """
        return self.om.make_tensor([data])

    def transform_action(self, action):
        """
        Remove the sequence axis for the environment.
        """
        transed_action = self.om.transform_action(action)
        transed_action = transed_action.squeeze(0)

        return transed_action


class ExperienceSequenceAgent(MethodWrapper):
    """
    An agent that inputs a sequence of experiences to the replay buffer instead
    of one at a time.
    """
    def __init__(self, agent, sequence_length, keep_length=0):
        """
        Args:
            agent (OffPolicyAgent): The agent to wrap.
            sequence_length (int): The length of the sequences.
            keep_length (int): Keeps the last n experiences from the previous
                               batch.
        """
        super().__init__(agent)

        self.sequence_length = sequence_length
        self.keep_length = keep_length

        # Store it in list format
        self.ready_experiences = {}
        self.num_experiences = 0

    def add_to_buffer(self, experience_queue, experiences, decay):
        """
        Adds the experience to the replay buffer.
        """
        experience = self._get_buffer_experience(experiences, decay)
        
        for key in experience:
            if key not in self.ready_experiences:
                self.ready_experiences[key] = []

            self.ready_experiences[key].append(experience[key])

        self.num_experiences += 1

        if self.num_experiences == self.sequence_length:
            # Concatenate experiences first
            experiences_to_send = {}
            for key in self.ready_experiences:
                experiences_to_send[key] = torch.cat(self.ready_experiences[key])

            experience_queue.put(experiences_to_send)
            keep_start = len(self.ready_experiences) - self.keep_length

            self.num_experiences = self.keep_length
            for key in self.ready_experiences:
                self.ready_experiences[key] = self.ready_experiences[key][keep_start:]

