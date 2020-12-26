from typing import Namespace

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from hlrl.core.envs.env import Env

class UnityEnv(Env):
    """
    A environment from Unity
    """
    def __init__(self, env: UnityEnvironment, flatten: bool = False):
        """
        Creates the given environment from Unity

        Args:
            env: The Unity environment to wrap.
            flatten: If the batch dimension of the observations should be
                flattened
        """
        Env.__init__(self)

        self.env = env
        self.flatten = flatten

        if flatten:
            self.state_space = np.sum(
                brain.vector_observation_space_size for brain in self.env.brains
            )
            self.action_space = np.sum(
                brain.vector_action_space_size for brain in self.env.brains
            )
        else:
            self.state_space = self.env.brains[0].vector_observation_space_size
            self.action_space = self.env.brains[0].vector_action_space_size

    def _update_env_state(self, env_state: Namespace):
        """
        Updates the wrapper propers from the environment state.
        
        Args:
            env_state: The namespace containing the state variables.
        """
        self.state = env_state.vector_observations
        self.reward = env_state.rewards
        self.terminal = env_state.local_done

        if self.flatten:
            self.state = np.reshape(
                self.state.shape[0] * self.state.shape[1], self.state.shape[2:]
            )

            self.reward = np.mean(self.reward, dim=0)
            self.terminal = np.prod(self.terminal, dim=0)

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

    def reset(self, train_mode=False):
        """
        Resets the environment.
        """
        self._update_env_state(self.env.reset(train_mode))

        return self.state
