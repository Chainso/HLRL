from typing import Any, Dict, Optional, Tuple, Union

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

    def reset(self) -> None:
        """
        Resets the current step of the agent.
        """
        self.current_step = 0
        self.om.reset()

    def transform_terminal(
            self,
            terminal: Union[int, bool, np.ndarray],
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
        terminal = self.om.transform_terminal(terminal, info)

        truncated = info.get("TimeLimit.truncated") or 0
        
        if self.max_steps is not None and self.current_step >= self.max_steps:
            truncated = truncated or (1 - terminal)
            info["TimeLimit.truncated"] = truncated

            # Make sure the set the environment terminal to reset properly
            self.env.terminal = True

        terminal = terminal * (1 - truncated)

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
