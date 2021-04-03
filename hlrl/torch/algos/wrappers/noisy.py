from typing import Any, Dict

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.torch.layers import NoisyLayer

class NoisyAlgo(MethodWrapper):
    """
    A wrapper to reset the noise of a noisy network after each training batch
    and on reset.
    """
    def train_processed_batch(
            self,
            rollouts: Dict[str, Any],
            *args: Any,
            **kwargs: Any
        ) -> Any:
        """
        Resets the noise after training.

        Args:
            rollouts: The training batch to process.
            args: Positional training arguments.
            kwargs: Keyword training arguments.

        Returns:
            The return value of the wrapped algorithm train batch.
        """
        for module in self.modules():
            if isinstance(module, NoisyLayer):
                module.reset_noise()

        training_ret = self.om.train_batch(*args, **kwargs)

        return training_ret
