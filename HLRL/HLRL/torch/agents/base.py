from abc import ABC

from HLRL.core.agents.base import RLAgent

class TorchRLAgent(RLAgent, ABC):
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def __init__(self, env, algo, logger=None):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            env (Env): The environment the agent will explore in.

            algo (TorchRLAlgo): The algorithm the agent will use the explore the
                           environment.
            logger (Logger, optional) : The logger to log results while
                                        interacting with the environment.
        """
        super(self).__init__(env, algo, logger)

    def step(self):
        """
        Takes 1 step in the agent's environment. Returns the
        (state, action, reward, next state, terminal, additional algo returns,
        env info) tuple. Resets the environment if the current state is a
        terminal.
        """
        if(self.env.terminal == True):
            self.env.reset()

        state = self.env.state
        algo_step = self.algo.step(state)
        action = algo_step[0]

        # Get additional information if it is there
        if(len(algo_step) > 1):
            add_algo_rets = algo_step[1:]
        else:
            add_algo_rets = []

        next_state, reward, terminal, info = self.env.step(action)

        return (state, action, reward, next_state, terminal, add_algo_rets,
                info)

    def play(self, num_episodes):
        """
        Resets and plays the environment with the algorithm. Returns the average
        reward per episode.

        Args:
            num_episodes (int): The numbers of episodes to play for.
        """
        avg_reward = 0

        for episode in range(1, num_episodes + 1):
            self.env.reset()
            ep_reward = 0

            while(self.env.terminal == False):
                ep_reward += self.step()[2]

            if(self.logger is not None):
                self.logger["Play"] = ep_reward, episode

            avg_reward += ep_reward / num_episodes

        if(self.logger is not None):
            self.logger["Play/Average"] = avg_reward

        return avg_reward
