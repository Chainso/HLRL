from typing import Any, Callable

class Logger(dict):
    """
    Logs values with a dictionary with a key and value given and calls a hooked
    function when a dictionary value is added or updated.
    """
    def __init__(self, on_update: Callable[[Any, Any], Any]):
        """
        Logs values by key and calls a given function when a key is set or
        updated.

        Args:
            on_update: A function to run with the key and value parameters every
                time a value is set.
        """
        super().__init__()

        self._on_update = on_update

    def __setitem__(self, key, value):
        # The value is likely to be a tuple of a number and step/episode/epoch
        # count
        super().__setitem__(key, value)
        self._on_update(key, value)
