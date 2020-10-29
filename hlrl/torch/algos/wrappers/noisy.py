from hlrl.core.common.wrappers import MethodWrapper

class NoisyAlgo(MethodWrapper):
    """
    A wrapper to reset the noise of a noisy network after each training batch
    and on reset.
    """
    def train_batch(self, *training_args):
        """
        Resets the noise after training.
        """
        for module in self.modules():
            if isinstance(module, NoisyLayer):
                module.reset_noise()

        training_ret = self.om.train_batch(*training_args)

        return training_ret