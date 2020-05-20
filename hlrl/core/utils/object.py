from types import MethodType, SimpleNamespace


class MethodWrapper():
    """
    Wraps an object so that it mimics class inheritance. To access the wrapped
    object's methods, use self.om instead of self.obj.
    """
    def __init__(self, obj):
        """
        Wraps the object given.

        Args:
            obj (object): The object to wrap.
        """
        self.obj = obj

        # Overwritten methods, kept in a dictionary because using self.obj will
        # result in in infinite recursion
        init_dict = {}
        if isinstance(obj, MethodWrapper):
            init_dict = vars(obj.om)

        self.om = SimpleNamespace(**init_dict)

        for attr_name in dir(self.obj):
            if not attr_name.startswith("__"):
                attr = getattr(self.obj, attr_name)

                if isinstance(attr, MethodType):
                    setattr(self.om, attr_name, attr)

        for attr_name in dir(self):
            if (attr_name not in dir(MethodWrapper) and attr_name != "obj"
                and attr_name != "om"):
                attr = getattr(self, attr_name)
                setattr(self.obj, attr_name, attr)

                    
    def __getattr__(self, name):
        """
        Performs a chain lookup with the object. If the attribute is a method,
        calls the method using this object as self instead of the wrapped
        object.
        """
        if "obj" not in vars(self):
            raise AttributeError("{0} + has no attribute {1}".format(type(self),
                                                                     name))

        return getattr(self.obj, name)
        
    def __setattr__(self, name, value):
        if name == "obj" or name == "om":
            object.__setattr__(self, name, value)
        else:
            setattr(self.obj, name, value)