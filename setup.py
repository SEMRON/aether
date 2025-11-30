import os
import glob
import re
from setuptools.command.build_py import build_py
from setuptools import setup
import importlib.util


def proto_compile(source_dir: str, output_dir: str):
    import grpc_tools.protoc

    spec = importlib.util.find_spec("hivemind")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Could not find hivemind module")
    hivemind_base_path = spec.submodule_search_locations[0]

    cli_args = [
        "grpc_tools.protoc",
        f"--proto_path={source_dir}",
        f"-I{hivemind_base_path}/proto",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
    ] + glob.glob(f"{source_dir}/*.proto")

    code = grpc_tools.protoc.main(cli_args)
    if code:  # hint: if you get this error in jupyter, run in console for richer error message
        raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")
    # Adjust the import of the used hivemind proto code
    for script in glob.iglob(f"{output_dir}/*.py"):
        with open(script, "r+") as file:
            code = file.read()
            file.seek(0)
            file.write(re.sub(r"\n(import .+_pb2.*)", "\nfrom hivemind.proto \\1", code))
            file.truncate()


class BuildProtoCommand(build_py):
    def run(self):
        source_dir = os.path.join("src", "distqat", "distributed", "proto")
        output_dir = os.path.join("src", "distqat", "distributed", "proto")
        proto_compile(source_dir, output_dir)
        super().run()

setup(
    cmdclass={
        'build_py': BuildProtoCommand,
    }
)
