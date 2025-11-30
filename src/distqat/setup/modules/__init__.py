import os
import importlib.util

def _load_module(name):
    """Load a module from <name>/module.py and import <name>Module from it."""
    # Get the directory of the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the module file
    module_path = os.path.join(base_dir, name, "module.py")

    # Load the module from the file
    spec = importlib.util.spec_from_file_location(f"{name}_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the specific class/function from the module
    module_class = getattr(module, f"{name}Module")

    return module_class

# Important: The order of the modules here also sets the order of operations in the scripts

allSetupModuleClasses = [_load_module(x) for x in [
    "Hostname",
    "CreateUser",
    "InstallUV",
    "GitHubKeys",
    "CloneKeys",
    "FirewallOpen",
    "RepoClone",
    "LocalSetup",
    "RepoSetup",
    "FinalRunner",
]]
