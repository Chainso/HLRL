import gym

from HLRL.core.envs.base import Env

class GymEnv(Env):
    """
    A environment from OpenAI Gym
    """
    def __init__(self, env_name, *env_args):
        """
        Creates the given environment from OpenAI Gym

        Args:
            env_name (str): The name of the environment to create

            env_args (list): Any additional arguments for the environment
        """
        self._gym = gym.make(env_name, env_args)

    def step(self, action):
        """
        Takes 1 step into the environment using the given action.

        Args:
            action (object): The action to take in the environment.
        """
        (self._state, self._reward, self._terminal,
         self._info) = self._gym.step(action)

        return self._state, self._reward, self._terminal, self._info

    def reset(self):
        """
        Resets the environment.
        """
        self._state = self._gym.reset()

        return self._state
