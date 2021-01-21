from typing import Any, Dict, List, Tuple

import torch

from hlrl.core.common.wrappers import MethodWrapper

class TorchOffPolicyAgent(MethodWrapper):
    """
    A torch off policy agent, meant to handle batching experiences to send to
    a replay buffer.
    """
    def create_batch(
            self,
            ready_experiences: Dict[str, List[Any]],
        ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Creates a batch of experiences to be trained on from the ready
        experiences.

        Args:
            ready_experiences: The experiences to be trained on.
        
        Returns:
            A dictionary of each field necessary for training.
        """
        batch, *rest = self.om.create_batch(ready_experiences)

        # Unsqueeze the batch dimension which was lost in the translation from
        # the rollout dictionary to a list of individual rollouts
        batch = tuple(
            {
                key: experience[key].unsqueeze(0) for key in experience
            }
            for experience in batch
        )

        return batch, *rest
