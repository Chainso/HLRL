from typing import Any, Tuple
from types import MethodType, SimpleNamespace


class MethodWrapper():
    """
    Wraps an object so that it mimics class inheritance. To access the wrapped
    object's methods, use self.om instead of self.obj.
    """
    def __init__(
            self,
            obj: Any
        ):
        """
        Wraps the object given.

        Args:
            obj: The object to wrap.
        """
        self.obj = obj

        # Overwritten methods, kept in a dictionary because using self.obj will
        # result in in infinite recursion
        init_dict = {}
        if isinstance(obj, MethodWrapper):
            init_dict = vars(obj.om)

        self.om = SimpleNamespace(**init_dict)

        for attr_name in object.__dir__(self.obj):
            if not attr_name.startswith("__"):
                attr = getattr(self.obj, attr_name)

                if isinstance(attr, MethodType):
                    setattr(self.om, attr_name, attr)

        for attr_name in dir(self):
            if (attr_name not in dir(MethodWrapper) and attr_name != "obj"
                and attr_name != "om"):
                attr = getattr(self, attr_name)
                setattr(self, attr_name, attr)

    def __getattr__(self,
                    name: str) -> Any:
        """
        Performs a chain lookup with the object.

        Args:
            name: The name of the attribute to look up.

        Returns:
            The value of the attribute.

        Raises:
            AttributeError: If the attribute is not present.
        """
        if "obj" not in vars(self):
            raise AttributeError("{0} + has no attribute {1}".format(type(self),
                                                                     name))

        return getattr(self.obj, name)
        
    def __setattr__(self,
                    name: str,
                    value: Any) -> None:
        """
        Sets the attribute to the lowest level wrapped object.
        
        Args:
            name: The name of the attribute to set.
            value: The value to set the attribute to.
        """
        if name == "obj" or name == "om" or name in dir(self):
            object.__setattr__(self, name, value)
        
        if name != "obj" and name != "om":
            setattr(self.obj, name, value)

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Returns a serialzed version of the wrapper on the object.

        Returns:
            The serialized wrapper.
        """
        return (type(self), (self.obj,))

    def __repr__(self) -> str:
        """
        Returns a string representation of the wrapper.
        
        Returns:
            The string representation of the wrapper.
        """
        return "{0}{{obj={1}}}".format(type(self).__name__, str(self.obj))