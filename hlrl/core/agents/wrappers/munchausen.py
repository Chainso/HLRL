from hlrl.core.common.wrappers import MethodWrapper

class MunchausenAgent(MethodWrapper):
    """
    An agent using Munchausen-RL from:
    https://arxiv.org/pdf/2007.14430.pdf
    """
    def __init__(self, agent, alpha):
        """
        Turns the agent into a Munchausen agent, assuming the underlying
        algorithm already has a temperature for their update
        """
        super().__init__(agent)

        self.alpha = alpha
        self.log_probs = 0

    def transform_algo_step(self, algo_step):
        """
        Updates the hidden state to the last output of the algorithm extras.
        """
        self.log_probs = algo_step[-1]

        return self.om.transform_algo_step(algo_step[:-1])

    def transform_reward(self, state, algo_step, reward, next_state):
        """
        Adds the Munchausen reward to the reward
        """
        return self.om.transform_reward(
            state, algo_step,
            reward + self.alpha * self.algo.temperature * self.log_probs,
            next_state
        )
