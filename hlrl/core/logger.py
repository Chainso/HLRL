from torch.utils.tensorboard import SummaryWriter

class Logger(dict):
    """
    Logs values with a dictionary with a key and value given and calls a hooked
    function when a dictionary value is added or updated
    """
    def __init__(self, on_update, *args, **kwargs):
        """
        Logs values by key and calls a given function when a key is set or
        updated.

        Args:
            on_update (function): A function to run with the key and value
                                  parameters every time a value is set.
            *args (list): All additional non-keyword arguments for a python
                          dictionary.
            **kwargs (list): All additional keyword arguments for a python
                             dictionary.
        """
        super().__init__(*args, **kwargs)

        self._on_update = on_update

    def __setitem__(self, key, value):
        # The value is likely to be a tuple of a number and step/episode/epoch
        # count
        super().__setitem__(key, value)
        self._on_update(key, value)

def make_tensorboard_logger(logs_path):
    """
    Creates a logger using tensorboardX summary writer and add scalar on update.

    Args:
        logs_path (str): The path to store the tensorboard logs in.
    """
    tensorboard = SummaryWriter(logs_path)

    # Make sure to deal with single values and tuple values
    def add_val(key, val):
        if(type(val) == tuple):
            tensorboard.add_scalar(key, *val)
        else:
            tensorboard.add_scalar(key, val, 1)

    return Logger(add_val)
    