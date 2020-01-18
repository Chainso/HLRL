from types import MethodType
from copy import copy

class MethodWrapper():
    """
    Wraps an object so that it mimics class inheritance.
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
        if "obj" not in vars(self):
            raise AttributeError("{} + has no attribute {}", type(self), name)

        attr = getattr(self.obj, name)

        if isinstance(attr, MethodType):
            # A shallow copy to replace obj in wrapper of wrapper to stop
            # infinite recursion
            binder = self
            if isinstance(self.obj, MethodWrapper):
                binder = copy(binder)
                binder.obj = self.obj.obj

            return binder.rebind_method(attr)
        else:
            return attr

    def _copy(self):
        """
        Copies everything, but sets the object to be this wrapped object's
        object if applicable.
        """
        
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
        return method.__func__.__get__(self, type(self))