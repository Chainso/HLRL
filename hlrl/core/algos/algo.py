from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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
    def create_optimizers(self) -> None:
        """
        Creates the optimizers for the algorithm, separate from the
        intialization so that the model can be moved to a different device first
        if needed.
        """
        raise NotImplementedError

    def process_batch(
            self,
            rollouts: Dict[str, Any],
        ) -> Dict[str, Any]:
        """
        Processes a batch to make it suitable for training.

        Args:
            rollouts: The training batch to process.

        Returns:
            The processed training batch.
        """
        return rollouts

    def train_batch(
            self,
            rollouts: Dict[str, Any],
            *args: Any,
            **kwargs: Any
        ) -> Any:
        """
        Processes a batch of rollouts then trains the network.

        Args:
            args: Positional training arguments.
            kwargs: Keyword training arguments.

        Returns:
            The training return on the batch.
        """
        processed_batch = self.process_batch(rollouts, *args, **kwargs)
        return self.train_processed_batch(*processed_batch)

    @abstractmethod
    def train_processed_batch(
            self,
            rollouts: Dict[str, Any],
            *args: Any,
            **kwargs: Any
        ) -> Any:
        """
        Trains the network for a batch of rollouts.

        Args:
            rollouts: The training batch to process.
            args: Positional training arguments.
            kwargs: Keyword training arguments.
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
    def save_dict(self) -> Dict[str, Any]:
        """
        Saves in the current state of the algorithm in a dictionary.

        Returns:
            A dictionary of values to save this algorithm.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path):
        """
        Saves the algorithm to a given save path.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dict(self, load_path):
        """
        Reads and returns the load dictionary from the load path.
        """
        raise NotImplementedError

    @abstractmethod
    def load(
            self,
            load_path: str = "",
            load_dict: Optional[Dict[str, Any]] = None
        ):
        """
        Loads the algorithm from a given save path. Will use the given state
        dictionary if given, or load from a file otherwise.
        """
        raise NotImplementedError

    def reset_hidden_state(self):
        """
        Resets the hidden state, if this is a recurrent algorithm.
        """
        raise NotImplementedError