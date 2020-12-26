from argparse import Namespace
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from mlagents_envs.environment import (
    UnityEnvironment, DecisionSteps, TerminalSteps, ActionTuple
)
from hlrl.core.envs.env import Env

class UnityEnv(Env):
    """
    A environment from Unity
    """
    def __init__(self,
                 env: UnityEnvironment,
                 behaviour_name: Optional[str] = None):
        """
        Creates a vectorized environment of the given environment from Unity

        Args:
            env: The Unity environment to wrap.
            behaviour_name: The behaviour name of the environment wrapper,
                defaults to the first behaviour in the spec list.
        """
        Env.__init__(self)

        self.env = env

        self.env.reset()

        self.behaviour_name = behaviour_name or list(self.env.behavior_specs)[0]
        self.spec = self.env.behavior_specs[self.behaviour_name]

        self.state_space = self.spec.observation_shapes[0]
        self.action_space = (self.spec.action_spec.continuous_size,)

    def _update_env_state(
            self
        ) -> Tuple[List[Any], List[Any], List[Union[bool, int]], List[Any]]:
        """
        Transforms a step batch into a standard (s', r, t, info) batch.

        Returns:
            A transition tuple for the batch of agents.
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behaviour_name)

        agent_ids = []
        self._state = []
        self.reward = []
        terminal = []
        self.info = []

        for agent_id in decision_steps:
            decision_step = decision_steps[agent_id]

            agent_ids.append(agent_id)
            self._state.append(decision_step.obs[0])
            self.reward.append([decision_step.reward])
            terminal.append([False])
            self.info.append([None])

        for agent_id in terminal_steps:
            terminal_step = terminal_steps[agent_id]

            agent_ids.append(agent_id)
            self._state.append(terminal_step.obs[0])
            self.reward.append([terminal_step.reward])
            terminal.append([not terminal_step.interrupted])
            self.info.append([None])

        sort_order = np.argsort(agent_ids)

        self._state = np.array(self._state)[sort_order]
        self.reward = np.array(self.reward)[sort_order]
        terminal = np.array(terminal)[sort_order]
        self.terminal = np.prod(terminal)
        self.info = np.array(self.info)[sort_order]

        return self._state, self.reward, terminal, self.info

    @property
    def state(self):
        """
        Returns the latest state from the environment.
        """
        self._update_env_state()

        return self._state

    @state.setter
    def state(self, state):
        """
        Sets the current state.

        Args:
            state: The state to set.
        """
        self._state = state

    def step(self, action: List[Tuple[float, ...]]):
        """
        Takes 1 step into the environment using the given action.

        Args:
            action: The action to take in the environment.
        """
        action = ActionTuple(continuous=action)
    
        self.env.set_actions(self.behaviour_name, action)
        self.env.step()

        return self._update_env_state()

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
        self.env.reset()
        self._update_env_state()

        return self._state

    def close(self):
        """
        Closes the Unity environment.
        """
        self.env.close()
