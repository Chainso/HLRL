from gym import ActionWrapper, Env, spaces

from hlrl.core.envs.gym.spaces import (
    DiscreteShaped, FlattenedTupleShaped, TupleShaped
)

class ShapedActionWrapper(ActionWrapper):
    """
    Changes the action space to be a tuple of (discrete, continuous).
    """
    def __init__(self, env: Env):
        """
        Wraps the environment to transform the action space to the tuple
        parameterized form.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)

        self.action_space = self._transform_action_space(env.action_space)

    def _transform_action_space(self, space):
        if isinstance(space, spaces.Discrete):
            space = DiscreteShaped(space.n)
        elif isinstance(space, spaces.Box):
            space = spaces.Box(
                space.low.flatten(), space.high.flatten(), shape=space.shape,
                dtype=space.dtype
            )
        elif (isinstance(space, spaces.MultiBinary)
            or isinstance(space, spaces.MultiDiscrete)):
            space = DiscreteShaped(spaces.flatdim(space))
        elif isinstance(space, spaces.Tuple):

            conts = []
            discretes = []

            space_iter = list(space.spaces)
            for sp in space_iter:
                if isinstance(sp, spaces.Tuple):
                    space_iter += sp.spaces
                else:
                    sp = self._transform_action_space(sp)

                    if isinstance(sp, spaces.Box):
                        conts.append(sp)
                    elif isinstance(sp, spaces.Discrete):
                        discretes.append(sp)

            cont_space = None
            discrete_space = None

            if len(conts) == 1:
                cont_space = conts[0]
            elif len(conts) > 1:
                cont_space = FlattenedTupleShaped(conts)

            if len(discretes) == 1:
                discrete_space = discretes[0]
            if len(discretes) > 1:
                discrete_space = FlattenedTupleShaped(discretes)

            if cont_space is None:
                space = discrete_space
            elif discrete_space is None:
                space = cont_space
            else:
                space = TupleShaped([discrete_space, cont_space])

            if isinstance(space, spaces.Tuple):
                space = TupleShaped(space.spaces)

        return space

    def action(self, action):
        """
        Transforms the action for the parameterized action wrapper.

        Args:
            action: The action to transform.

        Returns:
            The transformed, parameterized action.
        """
        return action
