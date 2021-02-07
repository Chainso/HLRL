from typing import Any, Tuple

from hlrl.core.common.wrappers import MethodWrapper

class VectorizedEnv(MethodWrapper):
    """
    A wrapper of vectorized environments, used to handle terminal steps
    properly.
    """
    def step(
            self,
            action: Tuple[Any]
        ) -> Tuple[Tuple[Any], Tuple[Any], Tuple[Any], Any]:
        """
        Takes 1 step into the environment using the given action.

        Args:
            action: The action to take in the environment.
        """
        state, reward, terminal, info = self.om.step(action)
        self.terminal = all(terminal)

        return state, reward, terminal, info
