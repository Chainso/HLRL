from hlrl.core.utils import MethodWrapper
import torch

class RecurrentAgent(MethodWrapper):
    """
    A recurrent agent that holds a hidden state.
    """
    def __init__(self, agent):
        """
        Turns the agent into a recurrent agent.
        """
        super().__init__(agent)

        self.hidden_state = None
        self.last_action = None

    def set_hidden_state(self, hidden_state):
        """
        Sets the hidden state to the given one.
        """
        self.hidden_state = hidden_state

    def transform_state(self, state):
        """
        Appends the hidden state to the algorithm inputs.
        """
        transed_state = self.om.transform_state(state)
        transed_state["last_action"] = self.last_action
        transed_state["hidden_state"] = self.hidden_state

        return transed_state

    def transform_action(self, action):
        """
        Updates the last action.
        """
        self.last_action = action
        return self.om.transform_action(action)

    def transform_algo_step(self, algo_step):
        """
        Updates the hidden state to the last output of the algorithm extras.
        """
        transed_algo_step = self.om.transform_algo_step(algo_step)
        transed_algo_step["next_hidden_state"] = algo_step[-1]

        return transed_algo_step

    def reset(self):
        """
        Resets the agent's hidden state
        """
        self.set_hidden_state(self.obj.algo.reset_hidden_state())
        self.last_action = self.make_tensor(self.env.sample_action())