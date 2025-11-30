from typing import Any

import importlib


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Load a Python object (e.g., class, function, variable) from a module path.

    The object path may be given as ``"module.submodule.ObjectName"``. If no module
    path is provided, ``default_obj_path`` will be used as the module path.

    Parameters
    ----------
    obj_path : str
        Path to the object to be loaded, including its name. If the path does not
        contain a module name, ``default_obj_path`` will be used.
    default_obj_path : str, optional
        Default module path to use when ``obj_path`` does not specify one.
        Default is an empty string.

    Returns
    -------
    Any
        The loaded Python object.

    Raises
    ------
    AttributeError
        If the module does not contain the specified object.
    ModuleNotFoundError
        If the module cannot be imported.

    Examples
    --------
    >>> load_obj("math.sqrt")
    <built-in function sqrt>
    >>> load_obj("sqrt", default_obj_path="math")
    <built-in function sqrt>
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)
