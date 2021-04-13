import numpy as np
from gym import ActionWrapper, Env, Space, spaces

from hlrl.core.envs.gym.spaces import FlattenedTupleShaped, TupleShaped

class RescaleParameterizedAction(ActionWrapper):
    """
    Rescales the continuous actions of a parameterized action space.
    """
    def __init__(self, env: Env, low: float, high: float):
        """
        Rescales the continuous actions of the parameterized action space to
        have the low and high given.

        Args:
            env: The environment with the action space to wrap.
            low: The infinum of the action space.
            hi
            gh: The suprenum of the action space.
        """
        discrete_space, continuous_space = env.action_space.spaces
        
        assert \
            (isinstance(continuous_space, spaces.Box) \
             or isinstance(continuous_space, spaces.Tuple)), \
            "expected Box or Tuple for continuous action space, got {}".format(
                type(continuous_space)
            )
        

        super().__init__(env)

        self.low = np.zeros(
            continuous_space.shape, dtype=continuous_space.dtype
        )
        self.low += low

        self.high = np.zeros(
            continuous_space.shape, dtype=continuous_space.dtype
        )
        self.high += high

        if isinstance(continuous_space, spaces.Tuple):
            continuous_space = tuple(
                spaces.Box(
                    low=low, high=high, shape=space.shape, dtype=space.dtype
                )
                for space in continuous_space
            )
            continuous_space = FlattenedTupleShaped(continuous_space)
        else:
            continuous_space = spaces.Box(
                low=low, high=high, shape=continuous_space.shape,
                dtype=continuous_space.dtype
            )

        self.action_space = TupleShaped([discrete_space, continuous_space])

    def _transform_box_action(
            self,
            action: np.ndarray,
            space: Space,
            low: float,
            high: float
        ) -> np.ndarray:
        """
        Transforms a box action to the proper bounds.

        Args:
            action: The box action to transform.
            space: The original untranslated space the action belongs to.
            low: The new low of the space.
            high: The new high of the space.

        Returns:
            The translated action.
        """
        cont_action = space.high - space.low
        cont_action *= ((cont_action - self.low) / (self.high - self.low))
        cont_action += space.low
        cont_action = np.clip(cont_action, space.low, space.high)

        action[-1] = cont_action
    def action(self, action):
        """
        Transforms the action to conform to the range of the wrapped
        environment.

        Args:
            action: The parameterized action to transform.

        Returns:
            The translated action.
        """
        cont_action = action[-1]

        assert np.all(np.greater_equal(cont_action, self.low)), \
               (action, self.low)

        assert np.all(np.less_equal(cont_action, self.high)), \
               (action, self.high)

        cont_space = self.env.action_space.spaces[-1]

        if isinstance(cont_space, spaces.Tuple):
            cont_action = np.array([
                self._transform_box_action(act, space)
                for act, space in zip(cont_action, cont_space)
            ])
        else:
            # Box action
            cont_action = self._transform_box_action(cont_action, cont_space)

        action[-1] = cont_action

        return action
