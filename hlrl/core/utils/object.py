from types import MethodType

class MethodWrapper():
    """
    Wraps an object so that it mimics class inheritance,
    """
    def __init__(self, obj):
        """
        Wraps the object given. Wrapper specific attributes must be prefixed
        by "_self_".

        Args:
            obj (object): The object to wrap.
        """
        self.obj = obj

    def __getattr__(self, name):
        """
        Performs a chain lookup with the object. If the attribute is a method,
        calls the method using this object as self instead of the wrapped
        object.
        """
        attr = getattr(self.obj, name)

        if isinstance(attr, MethodType):
            return self.rebind_method(attr)
        else:
            return attr

    def __setattr__(self, name, value):
        if (name == "obj" or hasattr(type(self), name)
            or name.startswith("_self_")):
            object.__setattr__(self, name, value)
        else:
            setattr(self.obj, name, value)

    def rebind_method(self, method):
        """
        Rebinds a method from another object to use this object.
        """
        return self.curry_func(method.__func__)

    def curry_func(self, func):
        """
        Curries self as the first argument of a function.

        Args:
            func (function): The function to curry.
        """
        def curried_func(*args):
            return func(self, *args)

        return curried_func