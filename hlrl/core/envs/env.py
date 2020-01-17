from abc import ABC, abstractmethod

class Env(ABC):
    """
    An environment that an agent can interact with.
    """
    def __init__(self):
        """
        Creates the environment and allows an agent to interact with it.

        Properties:
            state_space (tuple): The dimensions of the input state.
            action_space (tuple): The dimensions of the output actions.
            state (object): The current state of the environment.
            reward (float): The reward of the last action taken in the
                            environment.

            terminal (bool): True if the current state is a terminal state.

            info (object): Any additional information of the environment.
        """
        # The current information for the environment
        self._state_space = ()
        self._action_space = ()
        self._state = None
        self._reward = 0
        self._terminal = False
        self._info = None

    @property
    def state_space(self):
        """
        Returns the dimensions of the input state.
        """
        return self._state_space

    @property
    def action_space(self):
        """
        Returns the number of actions in the environment.
        """
        return self._action_space

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
    def step(self, action):
        """
        Takes 1 step into the environment using the given action.

        Args:
            action (object): The action to take in the environment.

        Returns an array of (next state, reward, terminal, info) tuples
        """
        raise NotImplementedError

    def render(self):
        """
        If applicable, the environment will render to the screen.
        """
        raise NotImplementedError

    def sample_action(self):
        """
        Samples an action from the environment action space.
        """
        raise NotImplementedError

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
        raise NotImplementedError
