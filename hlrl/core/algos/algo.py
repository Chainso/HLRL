from abc import ABC, abstractmethod

class RLAlgo(ABC):
    """
    An abstract reinforcement learning algorithm.
    """
    def __init__(self, logger=None):
        """
        Creates the reinforcement learning algorithm.

        Args:
            logger (Logger, optional): The logger to log results while training
                                       and evaluating.

        Properties:
            logger (Logger): The logger to log results while training and
                             evaluating.

            training_episodes (int): The number of episodes the algorithm
                                     has been training for.

            training_steps (int): The number of steps the algorithm has been
                                  training for.

            recurrent (bool): If the model is a recurrent model.

            env_steps (int): The number of environment steps the algorithm has
                             been training for
        """
        self.logger = logger

        self.env_episodes = 0
        self.training_steps = 0
        self.env_steps = 0

    @abstractmethod
    def train_batch(self, *train_args):
        """
        Trains the network for a batch of rollouts.

        Args:
            train_args (tuple) : The training arguments for the network.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, observation):
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation: A single observation from the environment.

        Returns:
            The action from for the observation given.
        """
        raise NotImplementedError

    @abstractmethod
    def save_dict(self):
        """
        Returns dictionary of values to save this algorithm.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path):
        """
        Saves the algorithm to a given save path.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, load_path):
        """
        Loads the algorithm from a given save path.
        """
        raise NotImplementedError

    def reset_hidden_state(self):
        """
        Resets the hidden state, if this is a recurrent algorithm.
        """
        raise NotImplementedError