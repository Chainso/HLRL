import functools

from typing import Any, Callable, Iterable, Iterator, Tuple

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
        functions: Tuple[Callable, ...],
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

class partial_iterator(functools.partial):
    """
    A partial function where keywords can be either a static keyword or a value
    acquired from an iterator.
    """
    def __call__(self, *args, **kwargs) -> Any:
        """
        Calls the underlying function, advance the iteration of all iterated
        keywords.

        Args:
            args: The arguments of the curried function call.
            kwargs: The keyword arguments of the curried function call.

        Returns:
            The value of the underlying function called with the given
                arguments.
        """
        generated_keywords = {}

        for keyword in self.keywords:
            value, generate = self.keywords[keyword]
            iterator_keys[keyword] = next(value) if generate else value

        all_keywords = {**self.keywords, **kwargs}

        return self.func(self.args, *args, **all_keywords)
