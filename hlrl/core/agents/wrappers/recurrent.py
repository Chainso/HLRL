from hlrl.core.utils import MethodWrapper

class RecurrentAgent(MethodWrapper):
    """
    A recurrent agent that holds a hidden state.
    """
    def __init__(self, agent):
        """
        Turns the agent into a recurrent agent.
        """
        super().__init__(agent)

        self.hidden_state = self.obj.algo.reset_hidden_state()
        self.last_action = self.env.sample_action()

    def set_hidden_state(self, hidden_state):
        """
        Sets the hidden state to the given one.
        """
        self.hidden_state = hidden_state

    def transform_state(self, state):
        """
        Appends the hidden state to the algorithm inputs.
        """
        return (*self.om.transform_state(state), self.last_action,
                self.hidden_state)

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
        algo_step = self.om.transform_algo_step(algo_step)
        self.hidden_state = algo_step[-1]

        return algo_step
