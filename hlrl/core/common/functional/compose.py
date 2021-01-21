import functools

from typing import Callable, Any, Tuple

def compose(inner_function: Callable, *functions: Callable):
    """
    Composes multiple functions f_n, f_{n - 1}, ..., f_1 into a single function
    f_n(f_{n - 1}(...(f_1(*args, **kwargs))))

    Args:
        inner_function (Callable): The innermost function.
        functions (Callable): A variable number of functions to compose.
    """
    return functools.partial(composed_functions, inner_function, functions)

def composed_functions(
        inner_function: Callable,
        functions: Tuple[Callable],
        *args: Any,
        **kwargs: Any
    ):
    """
    Returns the value of functions f_n(f_{n - 1}(...(f_1(*args, **kwargs))))

    Args:
        inner_function: The innermost function.
        functions: A variable number of functions to compose.
        *args: Arguments for the innermost function.
        **kwargs: Keyword arguments for the innermost function.
    """
    innermost_value = inner_function(*args, **kwargs)
    return functools.reduce(lambda x, f: f(x), functions, innermost_value)