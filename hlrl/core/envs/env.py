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
            truncated (bool): True if the current episode was truncated.
            info (object): Any additional information of the environment.
        """
        # The current information for the environment
        self.state_space = ()
        self.action_space = ()
        self.state = None
        self.reward = 0
        self.terminal = False
        self.truncated = False
        self.info = None

    @property
    def done(self):
        """
        Returns True if the environment episode is finished.
        """
        return self.terminal or self.truncated

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
