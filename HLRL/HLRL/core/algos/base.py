from abc import ABC, abstractmethod

class RLAlgo(ABC):
    """
    An abstract reinforcement learning algorithm
    """
    def __init__(self, logger=None):
        """
        Creates the reinforcement learning algorithm.

        Args:
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating.

        Properties:
            logger (Logger): The logger to log results while training and
                             evaluating.

            training_episodes (int): The number of episodes the algorithm
                                     has been training for.

            training_steps (int): The number of steps the algorithm has been
                                  training for.
        """
        self.logger = logger

        self.training_episodes = 0
        self.training_steps = 0

    def routine_save(self):
        """
        Checks to see if the number of steps done is a multiple of the save
        interval and will save the model if it is
        """
        if(self.steps_done % self.save_interval == 0 and self.training):
            self.save(self.save_path)

    @abstractmethod
    def start_training(self):
        """
        Starts training the network.
        """
        pass

    @abstractmethod
    def train_batch(self, rollouts):
        """
        Trains the network for a batch of rollouts.

        rollouts : The rollouts of training data for the network.
        """
        pass

    @abstractmethod
    def step(self, observation):
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation: A single observation from the environment.

        Returns:
            The action from for the observation given.
        """
        pass

    @abstractmethod
    def save(self, save_path):
        """
        Saves the algorithm to a given save path.
        """
        pass

    @abstractmethod
    def load(self, loads_path):
        """
        Loads the algorithm from a given save path.
        """
        pass
