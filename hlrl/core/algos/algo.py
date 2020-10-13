from abc import ABC, abstractmethod
from multiprocessing import Value
from typing import Generic

from hlrl.core.common.wrappers import MethodWrapper

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

            env_episodes (int): The number of environment episodes the algorithm
                                has been training for.

            env_steps (int): The number of environment steps the algorithm has
                             been training for.

            training_steps (int): The number of steps the algorithm has been
                                  training for.
        """
        self._logger = logger

        self._env_episodes = 0
        self._env_steps = 0
        self._training_steps = 0

    @property
    def logger(self):
        return self._logger

    @property
    def env_episodes(self):
        return self._env_episodes

    @property
    def env_steps(self):
        return self._env_steps

    @property
    def training_steps(self):
        return self._training_steps

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
    def load_dict(self, load_path):
        """
        Reads and returns the load dictionary from the load path.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, load_path, load_dict=None):
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

class DistributedAlgo(MethodWrapper):
    """
    A simple wrapper over an algorithm to update the basic algorithm properties
    in a distributed, thread-safe manner.
    """
    def __init__(self, algo: RLAlgo):
        """
        Creates the wrapper, using mp.Value to allow multiple processes to
        access the properties.
        """
        super().__init__(self, algo)

        old_env_episodes = self.env_episodes
        old_env_steps = self.env_steps
        old_training_steps = self.training_steps

        self._env_episodes = Value(int)
        self._env_episodes.value = old_env_episodes

        self._env_steps = Value(int)
        self._env_steps.value = old_env_steps

        self._training_steps = Value(int)
        self._training_steps.value = old_training_steps

    def _thread_safe_set(self, mp_value: Value[Generic[T]], new_value: T):
        """
        Sets the value by first acquiring the lock of the multiprocessing value
        then setting the value.

        Args:
            mp_value (Value[Generic[T]]): The multiprocessing value to set.
            new_value (T): The new value to set.
        """
        with mp_value.get_lock():
            mp_value.value = new_value

    @env_episodes.setter
    def env_episodes(self, episodes):
        self._thread_safe_set(self._env_episodes, episodes)

    @env_episodes.getter
    def env_episodes(self):
        return self._env_episodes.value

    @env_steps.setter
    def env_steps(self, steps):
        self._thread_safe_set(self._env_steps, steps)

    @env_steps.getter
    def env_steps(self):
        return self._env_steps.value

    @training_steps.setter
    def training_steps(self, steps):
        self._thread_safe_set(self._training_steps, steps)

    @training_steps.getter
    def training_steps(self):
        return self._training_steps.value