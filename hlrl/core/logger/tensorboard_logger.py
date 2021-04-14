from typing import Any, Iterable, Tuple, Union

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
        self.writer = SummaryWriter(logs_path)

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Returns a serialzed version of the Tensorboard logger.

        Returns:
            The serialized Tensorboard logger.
        """
        return (type(self), (self._logs_path,))

    def _add_val(self, key: str, val: Union[Any, Tuple[Any, int]]) -> None:
        """
        Adds a value or value-step pair to the writer to the graph with the key.

        Args:
            key: The key of the graph to add the value to.
            val: The value or value-step pair to add.
        """
        # Try to unpack values, otherwise just use default 1
        try:
            self.writer.add_scalar(key, *val)
        except:
            self.writer.add_scalar(key, val, 1)
