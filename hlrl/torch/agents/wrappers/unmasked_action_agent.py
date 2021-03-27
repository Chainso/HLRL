from typing import Any, Tuple

import torch
from hlrl.core.agents import RLAgent
from hlrl.core.common.wrappers import MethodWrapper


class UnmaskedActionAgent(MethodWrapper):
    """
    An agent that turns a masked action into it's unmasked form.
    """
    def __init__(
            self,
            agent: RLAgent,
            action_mask: Tuple[bool, ...],
            default_action: Tuple[float, ...]
        ):
        """
        Creates the unmasked action wrapper over the agent.

        Args:
            agent: The agent to wrap.
            action_mask: A mask that the given actions use.
            default_action: The default values for the action.
        """
        super().__init__(agent)

        if len(action_mask) != len(default_action):
            error_format = (
                "Action mask size {} and default action size {} "
                + "should be the same."
            )
            raise ValueError(
                error_format.format(len(action_mask), len(default_action))
            )

        self.action_mask = action_mask
        self.default_action = torch.tensor(default_action)

        action_mask_idx = torch.tensor(action_mask).nonzero(as_tuple=False)

        self.action_indices = action_mask_idx.transpose(0, 1)
        self.action_indices = self.action_indices.to(self.algo.device)

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Reduces the inputs used to serialize and recreate the unmasked action
        agent.

        Returns:
            A tuple of the class and input arguments.
        """
        return (type(self), (self.obj, self.action_mask, self.default_action))

    def transform_action(self, action: torch.Tensor):
        """
        Expands the action to it's unmasked form.

        Args:
            action: The action to take in the environment.

        Returns:
            The unmasked action.
        """
        expanded_mask = self.action_indices.expand(action.shape)
        expanded_default_action = self.default_action.expand(
            action.shape[:-1] + self.default_action.shape
        )

        unmasked_action = expanded_default_action.scatter(
            -1, expanded_mask, action
        )

        return self.om.transform_action(unmasked_action)
