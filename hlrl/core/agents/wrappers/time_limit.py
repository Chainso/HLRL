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

    def get_terminal_and_truncated(self, terminal: Any, truncated: Any, info: Any) -> tuple[Any, Any]:
        """
        Checks to see if the agent has terminated in the environment. An agent
        may not be terminated when an environment terminates in the case of
        time limits or other external factors.

        Args:
            terminal: The environment terminal value.
            truncated: The environment truncated value.
            info: Additional environment information for the step.

        Returns:
            True if the agent is in a terminal state
        """
        terminal, truncated = self.om.get_terminal_and_truncated(terminal, truncated, info)
        
        if self.max_steps is not None and self.current_step >= self.max_steps:
            truncated = True

        truncated = truncated * (1 - terminal)
        
        # Make sure the set the environment terminal to reset properly
        self.env.terminal = terminal or truncated

        return terminal, truncated

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
