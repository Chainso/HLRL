from collections import deque

from hlrl.core.utils import MethodWrapper
from hlrl.torch.agents import TorchRLAgent, OffPolicyAgent

class SequenceInputAgent(MethodWrapper, TorchRLAgent):
    """
    An agent that provides sequences of input to the model (of length 1).
    """
    def __init__(self, agent):
        MethodWrapper.__init__(self, agent)

    def make_tensor(self, data):
        """
        Creates a float tensor of the data of batch size 1.
        """
        agent_make_tens = self.rebind_method(self.obj.make_tensor)

        return agent_make_tens([data])

class ExperienceSequenceAgent(MethodWrapper, OffPolicyAgent):
    """
    An agent that inputs a sequence of experiences to the replay buffer instead
    of one at a time.
    """
    def __init__(self, agent, sequence_length):
        MethodWrapper.__init__(self, agent)

        self._self_sequence_length = sequence_length
        self._self_ready_experiences = []

    def _get_buffer_experience(self, experiences, decay):
        """
        Perpares the experience to add to the buffer.
        """
        reward = self._n_step_decay(experiences, decay)

        experience = experiences.pop()

        experience, algo_extras, next_algo_extras = experience
        q_val = algo_extras[0]
        next_q = next_algo_extras[0]

        experience[2] = reward

        target_q_val = reward + decay * next_q

        buffer_experience = (experience, q_val, target_q_val, *algo_extras[1:],
                             *next_algo_extras[1:])

        self._self_ready_experiences.append(buffer_experience)

    def add_to_buffer(self, experience_queue, experiences, decay):
        """
        Adds the experience to the replay buffer.
        """
        self._get_buffer_experience(experiences, decay)

        if len(self._self_ready_experiences) == self._self_sequence_length:
            experience_queue.put(self._self_ready_experiences)
            self._self_ready_experiences = []