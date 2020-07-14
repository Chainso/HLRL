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
        return self.om.transform_action(action)[0]

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

        self.ready_experiences = []
        self.q_vals = []
        self.target_q_vals = []
        self.a = True
        self.b = True
    def _get_buffer_experience(self, experiences, decay):
        """
        Perpares the experience to add to the buffer.
        """
        (experience, q_val, target_q_val,
            *extras) = self.om._get_buffer_experience(experiences, decay)

        buffer_experience = (experience, *extras)

        self.ready_experiences.append(buffer_experience)
        self.q_vals.append(q_val)
        self.target_q_vals.append(target_q_val)

    def add_to_buffer(self, experience_queue, experiences, decay):
        """
        Adds the experience to the replay buffer.
        """
        self._get_buffer_experience(experiences, decay)

        if (self.a or self.b) and len(self.ready_experiences) == self.sequence_length:
            q_vals = torch.cat(self.q_vals)
            target_q_vals = torch.cat(self.target_q_vals)
            print(self.ready_experiences[0][-1])
            print("+++++++++++++++++++++++++++++++++")
            print("Tryna put")
            experience_queue.put((self.ready_experiences, q_vals,
                                  target_q_vals))
            print("Put")
            keep_start = len(self.ready_experiences) - self.keep_length
            self.ready_experiences = self.ready_experiences[keep_start:]
            self.q_vals = self.q_vals[keep_start:]
            self.target_q_vals = self.target_q_vals[keep_start:]
            self.b = self.a
            self.a = False