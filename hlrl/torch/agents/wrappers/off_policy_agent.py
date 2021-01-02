from typing import Any, Dict, List, Tuple

from hlrl.core.common.wrappers import MethodWrapper

class TorchOffPolicyAgent(MethodWrapper):
    """
    A torch off policy agent, meant to handle batching experiences to send to
    a replay buffer.
    """
    def create_batch(
            self,
            ready_experiences: Dict[str, List[Any]],
        ) -> Tuple[Dict[str, Any]]:
        """
        Creates a batch of experiences to be trained on from the ready
        experiences.

        Args:
            ready_experiences: The experiences to be trained on.
        
        Returns:
            A dictionary of each field necessary for training.
        """
        batch = self.om.create_batch(ready_experiences)

        # Unsqueeze the batch dimension which was lost in the translation from
        # the rollout dictionary to a list of individual rollouts
        for experience in batch:
            for key in experience:
                experience[key] = experience[key].unsqueeze(0)

        return batch
