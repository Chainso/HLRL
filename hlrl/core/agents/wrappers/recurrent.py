from hlrl.core.utils import MethodWrapper
from hlrl.core.agents import RLAgent

class RecurrentAgent(MethodWrapper):
    """
    A recurrent agent that holds a hidden state.
    """
    def __init__(self, agent):
        """
        Turns the agent into a recurrent agent.
        """
        MethodWrapper.__init__(self, agent)

        self._self_hidden_state = self.algo.reset_hidden_state()
        self._self_last_action = self.env.sample_action()

    def set_hidden_state(self, hidden_state):
        """
        Sets the hidden state to the given one.
        """
        self._self_hidden_state = hidden_state

    def transform_state(self, state):
        """
        Appends the hidden state to the algorithm inputs.
        """
        print("------------------")
        print(self.obj.transform_state(state))
        print(self)
        return (*self.obj.transform_state(state), self._self_last_action,
                self._self_hidden_state)

    def transform_action(self, action):
        """
        Updates the last action.
        """
        self._self_last_action = action
        return self.obj.transform_action(action)

    def transform_algo_step(self, algo_step):
        """
        Updates the hidden state to the last output of the algorithm extras.
        """
        algo_step = self.obj.transform_algo_step(algo_step)
        self._self_hidden_state = algo_step[-1]

        return algo_step
