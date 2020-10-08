import functools

from typing import Callable, Any, Tuple

def compose(functions: Tuple[Callable]):
    """
    Composes multiple functions f_n, f_{n - 1}, ..., f_1 into a single function
    f_n(f_{n - 1}(...(f_1(*args, **kwargs))))
    """
    return functools.reduce(compose2, functions)

def compose2(func1: Callable, func2: Callable):
    """
    Composes 2 functions g, f into a single function f(g(*args, **kwargs)).

    Args:
        func1 (Callable): The outer function of the composition.
        func2 (Callable): The inner function of the composition.
    """
    def function_composition(*args: Any, **kwargs: Any):
        """
        Returns the result of the function composition.

        Args:
            *args (Any): The positional arguments to the inner function.
            **kwargs (Any): The keyword arguments to the inner function.
        """
        return func2(func1(*args, **kwargs))

    return function_composition