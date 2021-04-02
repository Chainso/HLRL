from typing import Any, Dict, Optional, Tuple

import numpy as np

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.core.agents import RLAgent

class TimeLimitAgent(MethodWrapper):
    """ 
    An agent with a set time limit.
    """
    def __init__(self, agent: RLAgent, max_steps: Optional[int] = None):
        """
        Creates the agent with a set time limit, that also handles the terminals
        relating to the time limit.

        Args:
            max_steps: The maximum number of steps per episode.
        """
        super().__init__(agent)

        self.max_steps = max_steps
        self.current_step = 0

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Reduces the inputs used to serialize and recreate the time limit agent.

        Returns:
            A tuple of the class and input arguments.
        """
        return (type(self), (self.obj, self.max_steps))

    def reset(self) -> None:
        """
        Resets the current step of the agent.
        """
        self.current_step = 0
        self.om.reset()

    def transform_terminal(
            self,
            terminal: int | bool | np.array,
            info: Dict[str, Any]
        ) -> Any:
        """
        Transforms the terminal of an environment step.

        Args:
            terminal: The terminal value to transform.
            info: Additional environment information for the step.

        Returns:
            The transformed terminal.
        """
        truncated = info.get("TimeLimit.truncated")
        
        if self.max_steps is not None and self.current_step >= self.max_steps:
            info["TimeLimit.truncated"] = truncated or (1 - terminal)

        terminal *= not info.get("TimeLimit.truncated")

        return terminal

    def step(self, *args: Any, **kwargs: Any) -> None:
        """
        Takes 1 step in the agent's environment. Returns the experience
        dictionary. Resets the environment if the current state is a
        terminal.

        Args:
            args: Positional arguments for the wrapped step function.
            kwargs: Keyword arguments for the wrapped step function.
        """
        self.current_step += 1
        return self.om.step(*args, **kwargs)
