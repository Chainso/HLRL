from typing import Namespace

from mlagents_envs.environment import UnityEnvironment
from hlrl.core.envs.env import Env

class UnityEnv(Env):
    """
    A environment from Unity
    """
    def __init__(self, env: UnityEnvironment):
        """
        Creates the given environment from Unity

        Args:
            env: The Unity environment to wrap.
        """
        Env.__init__(self)
        self.env = env
        self.state_space = self.env.observation_space.shape

        #self.action_space = self.env.action_space
        #self.action_space = (
            #(self.env.action_space.n,)
            #if isinstance(self.env.action_space, Discrete)
            #else self.env.action_space.shape
        #)

    def _update_env_state(self, env_state: Namespace):
        """
        Updates the wrapper propers from the environment state.
        
        Args:
            env_state: The namespace containing the state variables.
        """
        self.state = env_state.observations
        self.reward = env_state.rewards
        self.terminal = env_state.local_done

    def step(self, action: object):
        """
        Takes 1 step into the environment using the given action.

        Args:
            action: The action to take in the environment.
        """
        self._update_env_state(self.env.step(action))

        return self.state, self.reward, self.terminal, self.info

    def sample_action(self):
        return self.env.action_space.sample()

    def render(self):
        """
        Renders the Unity environment.
        """
        pass

    def reset(self):
        """
        Resets the environment.
        """
        self._update_env_state(self.env.reset())

        return self.state
