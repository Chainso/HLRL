from torch.utils.tensorboard import SummaryWriter

from .logger import Logger

class TensorboardLogger(Logger):
    """
    Logs values using tensorboard.
    """
    def __init__(self, logs_path: str):
        """
        Logs values by key and calls a given function when a key is set or
        updated.

        Args:
            logs_path: The path of the directory of the tensorboard logs.
        """
        super().__init__(self._add_val)

        self._logs_path = logs_path
        self._tensorboard = SummaryWriter(logs_path)

    def __reduce__(self):
        return (TensorboardLogger, (self._logs_path,))

    # Make sure to deal with single values and tuple values
    def _add_val(self, key, val):
        if(type(val) == tuple):
            self._tensorboard.add_scalar(key, *val)
        else:
            self._tensorboard.add_scalar(key, val, 1)
