from abc import ABC, abstractmethod

class Env(ABC):
    """
    An environment that an agent can interact with.
    """
    def __init__(self):
        """
        Creates the environment and allows an agent to interact with it.

        Properties:
            state (object): The current state of the environment.

            reward (float): The reward of the last action taken in the
                            environment.

            terminal (bool): True if the current state is a terminal state.

            info (object): Any additional information of the environment.
        """
        # The current information for the environment
        self._state = None
        self._reward = 0
        self._terminal = True
        self._info = None

    @property
    def state(self):
        """
        Returns the current state of the environment.
        """
        return self._state

    @property
    def reward(self):
        """
        Returns the reward of the last action taken in the environment.
        """
        return self._reward

    @property
    def terminal(self):
        """
        Returns true if the current state of the environment is a terminal
        state.
        """
        return self._terminal

    @property
    def info(self):
        """
        Returns any additional information about the environment.
        """
        return self._info

    @abstractmethod
    def step(self, action, repeat=1):
        """
        Takes 1 step into the environment using the given action.

        Args:
            action (object): The action to take in the environment.

        If repeat = 1, returns (next state, reward, terminal, info) else returns
        an array of (next state, reward, terminal, info) tuples
        """
        pass

    def n_steps(self, actions):
        """
        Takes 1 step into the environment using the given action.

        Args:
            action (object): The action to take in the environment.

        Returns:
            An array of (next state, reward, terminal, info) tuples
        """
        return [self.step(action) for action in actions]

    @abstractmethod
    def reset(self):
        """
        Resets the environment.
        """
        pass
