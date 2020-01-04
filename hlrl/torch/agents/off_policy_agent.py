from .agent import TorchRLAgent

class OffPolicyAgent(TorchRLAgent):
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def __init__(self, env, algo, experience_replay, render=False, logger=None,
                 device="cpu"):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            env (Env): The environment the agent will explore in.
            algo (TorchRLAlgo): The algorithm the agent will use the explore the
                                environment.
            experience_replay (ExperienceReplay): The experience replay to store
                                                  (state, action, reward,
                                                   next state) tuples in.
            render (bool): If the environment is to be rendered (if applicable).
            logger (Logger, optional) : The logger to log results while
                                        interacting with the environment.
            device (str): The device for the agent to run on.
        """
        super().__init__(env, algo, render, logger, device)
        self.experience_replay = experience_replay

    def train(self, num_episodes):
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            num_episodes (int): The number of episodes to train for.
        """
        for episode in range(1, num_episodes + 1):
            ep_reward = 0

            while(self.env.terminal == False):
                (state, action, reward, next_state, terminal, add_algo_rets,
                 info) = self.step()

                if(len(add_algo_rets) == 0):
                    next_algo_rets = []
                else:
                    next_algo_rets = self.algo(self.env.state)[1:]
                    if(terminal):
                        next_algo_rets[0] = 0

                self.experience_replay.add((state, action, reward, next_state,
                                            terminal), *add_algo_rets,
                                            *next_algo_rets)

                ep_reward += reward
                self.algo.training_steps += 1

            if(self.logger is not None):
                self.logger["Train/Epsiode Reward"] = ep_reward, episode

            self.algo.training_episodes += 1
