import functools

from typing import Callable, Any, Tuple

def compose(*functions: Callable):
    """
    Composes multiple functions f_n, f_{n - 1}, ..., f_1 into a single function
    f_n(f_{n - 1}(...(f_1(*args, **kwargs))))

    Args:
        functions (Callable): A variable number of functions to compose.
    """
    return functools.reduce(lambda x, f: f(x), functions)