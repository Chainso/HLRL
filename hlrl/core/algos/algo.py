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

class RLAlgoWrapper(RLAlgo):
    """
    A wrapper around an algorithm.
    """
    def __init__(self, algo):
        self.algo = algo

    def __getattr__(self, name):
        if name in vars(self.algo):
            return getattr(self.algo, vars)

    def train_batch(self, *training_args):
        return self.algo.train_batch(*training_args)

    def step(self, observation):
        return self.algo.step(observation)

    def save_dict(self):
        return self.algo.save_dict()

    def save(self, save_path):
        return self.algo.save(save_path)

    def load(self, load_path):
        return self.algo.load(load_path)