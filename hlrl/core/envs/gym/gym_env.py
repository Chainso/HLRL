from gym.spaces import Discrete

from hlrl.core.envs.env import Env

class GymEnv(Env):
    """
    A environment from OpenAI Gym
    """
    def __init__(self, env):
        """
        Creates the given environment from OpenAI Gym

        Args:
            env (gym.env): The name of the environment to create
        """
        Env.__init__(self)
        self.env = env
        self._state_space = self.env.observation_space.shape

        # Easier to just sample an action than deal with the different action
        # types of gym
        self._action_space = self.env.action_space
        self._action_space = (
            (self.env.action_space.n,)
            if isinstance(self.env.action_space, Discrete)
            else self.env.action_space.shape
        )

    def step(self, action):
        """
        Takes 1 step into the environment using the given action.

        Args:
            action (object): The action to take in the environment.
        """
        (self._state, self._reward, self._terminal,
         self._info) = self.env.step(action)

        return self._state, self._reward, self._terminal, self._info

    def sample_action(self):
        return self.env.action_space.sample()

    def render(self):
        """
        Renders the gym environment.
        """
        self.env.render()

    def reset(self):
        """
        Resets the environment.
        """
        self._state = self.env.reset()
        self._reward = 0
        self._terminal = False

        return self._state
