from abc import ABC, abstractmethod

class RLAgent(ABC):
    """
    An agent that interacts with an environment
    """
    def __init__(self, env, algo, logger=None):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            env (Env): The environment the agent will explore in.

            algo (RLAlgo): The algorithm the agent will use the explore the
                           environment.
            logger (Logger, optional) : The logger to log results while
                                        interacting with the environment.

        Properties:
            env (Env): The environment the agent explores in.

            algo (RLAlgo): The algorithm the agent uses to explore.

            logger (Logger): The logger the agent uses to record exploration
                             data.
        """
        self.env = env
        self.algo = algo
        self.logger = logger

    @abstractmethod
    def step(self):
        """
        Takes 1 step in the agent's environment.
        """
        pass

    @abstractmethod
    def play(self, num_episodes):
        """
        Plays the environment with the algorithm.

        Args:
            num_episodes (int): The numbers of episodes to play for.
        """
        pass

    @abstractmethod
    def train(self, num_episodes):
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            num_episodes (int): The number of episodes to train for.
        """
        pass
