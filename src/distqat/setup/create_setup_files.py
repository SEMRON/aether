#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Dict, Any, Tuple
import argparse
import json
import os
import stat
import subprocess
import sys
import textwrap
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, List, Optional, Protocol, Dict, Any
from pathlib import Path
import yaml

from .common import *
from .machine_info import *
from .modules import *
from .validation import *

# ------ System Dependencies ------

def get_system_dependencies(cfg: Config) -> Iterable[str]:
    pkgs = []
    if cfg.os.id in [Distro.UBUNTU, Distro.DEBIAN]:
        pkgs.extend([
            "qemu-guest-agent",
            "python3",
            "python3-pip"
        ])
    elif cfg.os.platform_id in [Platform.ENTERPRISE_LINUX_8, Platform.ENTERPRISE_LINUX_9]:
        pkgs.extend([
            "qemu-guest-agent",
            "python3",
            "python3-pip"
        ])
    else:
        raise RuntimeError(f"Don't know how to install system dependencies for {str(cfg.os)}")

    return pkgs

# ------------ Emitters ------------

def _indent_block(s: str, n: int) -> str:
    pad = " " * n
    lines = s.splitlines()
    indented = "\n".join((pad + line) if len(line) else "" for line in lines)

    return indented


def emit_cloud_init(cfg: Config, modules: List[Module]) -> str:
    if cfg.username == cfg.management_user:
        raise RuntimeError("Management user cannot be the same as the service user")

    if not cfg.authorized_pubkey:
        raise RuntimeError(
            "Trying to create cloud-init script, but did not provide any public key for management. "
            "This would mean that you wouldn't be able to manage the machine."
        )

    # Build cloud-init data structure
    cloud_init_data = {}

    # Users
    users = []
    service_user = {
        'name': cfg.username,
        'gecos': 'Aether service user',
        'shell': '/bin/bash',
    }
    if cfg.authorized_pubkey:
        service_user['ssh_authorized_keys'] = cfg.authorized_pubkey
    if cfg.management_user is None:
        service_user['sudo'] = "ALL=(ALL) NOPASSWD:ALL"
    users.append(service_user)

    if cfg.management_user is not None:
        management_user_dict = {
            'name': cfg.management_user,
            'gecos': 'Management user',
            'shell': '/bin/bash',
            'sudo': "ALL=(ALL) NOPASSWD:ALL",
        }
        if cfg.authorized_pubkey:
            management_user_dict['ssh_authorized_keys'] = cfg.authorized_pubkey
        users.append(management_user_dict)

    cloud_init_data['users'] = users

    # Package management
    cloud_init_data['package_update'] = True

    packages = get_system_dependencies(cfg)
    if packages:
        cloud_init_data['packages'] = packages

    # Collect files and commands from modules
    write_files = []
    runcmd = []
    set_owner_commands = []

    for m in modules:
        for f in m.files(cfg):
            write_files.append({
                'path': str(f.path),
                'permissions': f.mode,
                'owner': "root:root",
                'content': f.content.rstrip("\n") + "\n"
            })
            if f.user != "root" or f.group != "root":
                set_owner_commands.append(f"chown -R {f.owner()} {f.path}")
        runcmd.append(f"echo \"### Now running: {m.name}\"")
        runcmd.extend(m.commands(cfg))

    if write_files:
        cloud_init_data['write_files'] = write_files

    if runcmd or set_owner_commands:
        cloud_init_data['runcmd'] = set_owner_commands + runcmd

    class CloudInitDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    def str_presenter(dumper, data):
        """Force literal block style (|) for multiline strings"""
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    CloudInitDumper.add_representer(str, str_presenter)


    # Serialize to YAML with cloud-config header
    yaml_content = yaml.dump(
        cloud_init_data,
        Dumper=CloudInitDumper,
        default_flow_style=False,
        sort_keys=False,
        width=2**30
    )

    return "#cloud-config\n" + yaml_content


def emit_bash_script(cfg: Config, modules: List[Module], guard: bool = True) -> str:
    # Emit a single bash script that writes files and runs commands
    lines: List[str] = ["#!/usr/bin/env bash"]

    if guard:
        lines.extend([
            "install_script() {",
            "set -euo pipefail",
            "trap 'echo \"Part of the setup script failed, not contunuing with the rest\"; return 0' ERR"
        ])
    else:
        lines.append("set -euo pipefail")

    lines.extend(textwrap.dedent("""
        # Check if running as root
        if [[ $EUID -ne 0 ]]; then
            echo "This script must be run as root" >&2
            exit 1
        fi
    """).strip().split("\n"))

    # Write files via here-docs
    for f in (fs for m in modules for fs in m.files(cfg)):
        content = f.content.rstrip()
        eof_sequence = "EOF"

        # Check if EOF sequence appears in content with whitespace
        if f'\n{eof_sequence}\n' in f'\n{content}\n' or f' {eof_sequence} ' in f' {content} ':
            raise RuntimeError(f"EOF sequence '{eof_sequence}' found in content with surrounding whitespace for file {f.path}")

        lines.append(f"install -d -m {f.mode} -o {f.user} -g {f.group} $(dirname '{f.path}')")
        lines.append(
        f"cat > '{f.path}' <<'{eof_sequence}'\n"
        + content
        + f"\n{eof_sequence}"
        )
        lines.append(f"chmod {f.mode} '{f.path}'")
        lines.append(f"chown {f.user}:{f.group} '{f.path}'\n")

    # Commands
    for m in modules:
        lines.append("")
        lines.append(f"# Module: {m.name}")
        lines.append(f"echo \"### Now running: {m.name}\"")
        for cmd in m.commands(cfg):
            lines.append(cmd)

    if guard:
        lines.extend(textwrap.dedent("""
            # end of the guard
            }
            if ! (install_script); then
                echo "Install script failed"
                if [[ -t 0 ]] && [[ -t 1 ]]; then
                    read -p "Would you like to try running with sudo? (y/n): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        echo "Attempting to run with sudo..."
                        sudo bash -c "$(declare -f install_script); install_script"
                    else
                        echo "Skipping sudo attempt"
                    fi
                else
                    echo "Not on interactive shell, skipping sudo attempt"
                fi
            fi
        """).strip().split("\n"))

    return "\n".join(lines) + "\n"


def emit_dir_with_runner(cfg: Config, modules: List[Module], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    files_dir = outdir / "files"
    files_dir.mkdir(exist_ok=True)

    # Write file specs into a fake root hierarchy under files/
    file_specs: List[FileSpec] = list(fs for m in modules for fs in m.files(cfg))

    # Create the fake root hierarchy
    for f in file_specs:
        # Remove leading slash to create relative path
        relative_path = str(f.path).lstrip('/')
        fake_file_path = files_dir / relative_path

        # Create parent directories
        fake_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file content
        fake_file_path.write_text(f.content)

    # Generate runner script with hard-coded install commands
    runner = "#!/usr/bin/env bash" + textwrap.dedent("""
    set -euo pipefail

    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        echo "This script must be run as root" >&2
        exit 1
    fi

    ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
    FILES_DIR="$ROOT_DIR/files"

    # Install files from fake root to real locations
    """)

    # Add install commands for each file
    for f in file_specs:
        relative_path = str(f.path).lstrip('/')
        runner += f'install -D -m {f.mode} -o {f.user} -g {f.group} "$FILES_DIR/{relative_path}" "{f.path}"\n'

    runner += "\n# Module commands\n"

    cmd_lines = []
    for m in modules:
        cmd_lines.append("")
        cmd_lines.append(f"# Module: {m.name}")
        cmd_lines.append(f"echo \"### Now running: {m.name}\"")
        cmd_lines.extend(m.commands(cfg))
    runner += "\n".join(cmd_lines) + "\n"
    (outdir / "runner.sh").write_text(runner)
    os.chmod(outdir / "runner.sh", os.stat(outdir / "runner.sh").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def emit_skypilot_launch(cfg: Config, modules: List[Module]) -> str:
    # Ports from cfg
    ports = []
    for p in cfg.open_ports:
        rng = p.split('/')[0]
        ports.append(rng.replace(":", "-"))

    file_dump_cmds = []
    setup_cmds = []

    # Collect setup and run commands from modules
    for m in modules:
        # Add module files
        for f in m.files(cfg):
            # Write file content to a temp location then move it
            content = f.content.rstrip("\n")
            file_dump_cmds.append(f"cat > '/tmp/skysetup_{f.path.name}' <<'EOF'\n{content}\nEOF")
            file_dump_cmds.append(f"sudo install -D -m {f.mode} -o {f.user} -g {f.group} '/tmp/skysetup_{f.path.name}' '{f.path}'")
            file_dump_cmds.append("")

        # Add module commands to setup
        setup_cmds.append(f"echo '### Setup: {m.name}'")
        cmds = m.commands(cfg)
        if cmds:
            # Escape single quotes in commands and join with semicolons
            escaped_cmds = [cmd.replace("'", "'\\''") for cmd in cmds]
            combined_cmd = "; ".join(escaped_cmds)
            setup_cmds.append(f"sudo bash -c '{combined_cmd}'")
            setup_cmds.append("")

    # Build the YAML structure
    config = {
        "resources": {
            "ports": ports if ports else None
        },
        "setup": "\n".join(file_dump_cmds + setup_cmds),
        "run": "echo 'System configured successfully'"
    }

    # Remove None values
    config["resources"] = {k: v for k, v in config["resources"].items() if v is not None}
    if not config["resources"]:
        del config["resources"]

    class SkyPilotDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    def str_presenter(dumper, data):
        """Force literal block style (|) for multiline strings"""
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    SkyPilotDumper.add_representer(str, str_presenter)

    return yaml.dump(
        config,
        Dumper=SkyPilotDumper,
        default_flow_style=False,
        sort_keys=False,
        width=2**30
    )

# ------------ CLI ------------

GET_MACHINE_INFO_SCRIPT: Path = Path(__file__).parent / "scripts" / "get-initial-machine-info.sh"

def read_until_double_newline() -> str:
    """Read lines from stdin until two consecutive newlines are encountered."""
    lines = []
    blank_line_count = 0

    while True:
        try:
            line = input()
            if line == "":
                blank_line_count += 1
                if blank_line_count >= 2:
                    break
            else:
                blank_line_count = 0
            lines.append(line)
        except EOFError:
            break

    return "\n".join(lines)

def load_config(
        machine_info: Dict[str, Any],
        clone_key_path: Path|None,
        node_login_authorized_keys: List[str],
        source_url: str|None,
        source_ref: str|None,
        sources_target_dir: str|None,
        commands: List[str],
        user_override: str|None,
        management_user: str|None,
        gpu_vendor: GPU_VENDOR|None,
        install_drivers: bool|None) -> Config:
    os = get_os_from_machine_info(machine_info)
    if not gpu_vendor:
        gpu_vendor = get_gpu_vendor_from_machine_info(machine_info)

    if install_drivers is None:
        install_drivers = gpu_vendor is not None

    username = None
    if user_override:
        username = user_override
    if not username:
        mi_un = machine_info["user_info"]["username"]
        if mi_un:
            username = mi_un
        else:
            raise RuntimeError("Username neither provided explicitly, nor through machine info")

    # Load clone keys if provided
    deploy_key_private = None
    deploy_key_public = None

    if clone_key_path and clone_key_path.exists():
        try:
            deploy_key_private = clone_key_path.read_text()

            # Try to find the public key by appending .pub
            public_key_path = Path(str(clone_key_path) + ".pub")
            if public_key_path.exists():
                deploy_key_public = public_key_path.read_text()
            else:
                deploy_key_public = None
        except Exception as e:
            raise RuntimeError("Failed to read clone keys: " + str(e))

    return Config(
        os=os,
        source_url=source_url,
        source_ref=source_ref,
        sources_target_dir=sources_target_dir,
        username=username,
        management_user=management_user,
        gpu_vendor=gpu_vendor,
        create_new_user=bool(user_override),
        deploy_key_private=deploy_key_private,
        deploy_key_public=deploy_key_public,
        authorized_pubkey=node_login_authorized_keys,
        final_commands=commands,
        open_ports=[],
        install_drivers=install_drivers,
        extras={},
    )

def update_config(
        cfg: Config,
        clone_key_path: Path|None,
        node_login_authorized_keys: List[str],
        source_url: str|None,
        source_ref: str|None,
        sources_target_dir: str|None,
        commands: List[str],
        user_override: str|None,
        management_user: str|None,
        gpu_vendor: GPU_VENDOR|None,
        install_drivers: bool|None) -> Config:
    """
    Update an existing config with provided arguments.

    For any argument that is provided (non-empty list or not None),
    the corresponding field in the config is replaced.

    Returns a new Config object with the updates applied.
    """
    # Load clone keys if provided
    deploy_key_private = cfg.deploy_key_private
    deploy_key_public = cfg.deploy_key_public
    if clone_key_path is not None and clone_key_path.exists():
        try:
            deploy_key_private = clone_key_path.read_text()
            public_key_path = Path(str(clone_key_path) + ".pub")
            if public_key_path.exists():
                deploy_key_public = public_key_path.read_text()
        except Exception as e:
            raise RuntimeError("Failed to read clone keys: " + str(e))

    # Create and return new Config with updates applied where provided
    return Config(
        os=cfg.os,
        username=user_override if user_override is not None else cfg.username,
        management_user=management_user if management_user is not None else cfg.management_user,
        create_new_user=True if user_override is not None else cfg.create_new_user,
        gpu_vendor=gpu_vendor if gpu_vendor is not None else cfg.gpu_vendor,
        source_url=source_url if source_url is not None else cfg.source_url,
        source_ref=source_ref if source_ref is not None else cfg.source_ref,
        sources_target_dir=sources_target_dir if sources_target_dir is not None else cfg.sources_target_dir,
        final_commands=commands if commands else cfg.final_commands,
        deploy_key_private=deploy_key_private,
        deploy_key_public=deploy_key_public,
        authorized_pubkey=node_login_authorized_keys if node_login_authorized_keys else cfg.authorized_pubkey,
        open_ports=cfg.open_ports,
        install_drivers=install_drivers if install_drivers is not None else cfg.install_drivers,
        extras=cfg.extras
    )

def get_upstream_repo_url_and_ref_for_pwd(private_key_path: Optional[Path] = None) -> Tuple[str, str]:
    """
    Get the URL of the upstream repository for the current branch.

    Args:
        private_key_path: Optional path to a private key file to test access

    Returns:
        The URL of the upstream repository

    Raises:
        RuntimeError: If no upstream repo exists or access check fails
    """

    base_error_message = "Could not auto detect repo git URL: "

    # Check if we're in a git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            check=True,
            text=True
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(base_error_message + "Not in a git repository")

    # Get current branch name
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True
        )
        current_branch = result.stdout.strip()
    except subprocess.CalledProcessError:
        raise RuntimeError(base_error_message + "Could not determine current branch (detached HEAD?)")

    # Get upstream branch
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", f"{current_branch}@{{upstream}}"],
            capture_output=True,
            check=True,
            text=True
        )
        upstream_branch = result.stdout.strip()
    except subprocess.CalledProcessError:
        raise RuntimeError(base_error_message + f"Branch '{current_branch}' has no upstream branch configured")

    # Extract remote name from upstream branch (e.g., "origin/main" -> "origin")
    remote_name = upstream_branch.split('/')[0]

    # Get remote URL
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", remote_name],
            capture_output=True,
            check=True,
            text=True
        )
        repo_url = result.stdout.strip()
    except subprocess.CalledProcessError:
        raise RuntimeError(base_error_message + f"Could not get URL for remote '{remote_name}'")

    if private_key_path and not private_key_path.exists():
        raise RuntimeError(base_error_message + f"Private key file does not exist: {private_key_path}")

    # Get current commit as the ref
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True
        )
        current_ref = result.stdout.strip()
    except subprocess.CalledProcessError:
        raise RuntimeError(base_error_message + "Could not determine current commit")

    return (repo_url, current_ref)

def get_repo_name_from_git_url(url: str|None) -> str|None:
    """
    Extract the repository name from a git URL.

    Handles various git URL formats:
    - https://github.com/user/repo.git
    - git@github.com:user/repo.git
    - ssh://git@github.com/user/repo.git
    - https://gitlab.com/user/repo

    Args:
        url: The git repository URL

    Returns:
        The repository name (without .git extension)

    Raises:
        ValueError: If the URL format cannot be parsed
    """
    if not url:
        return None

    # Remove trailing slashes
    url = url.rstrip('/')

    # Handle SSH URLs (git@host:path or ssh://git@host/path)
    if url.startswith('ssh://'):
        # ssh://git@github.com/user/repo.git -> user/repo.git
        url = url.replace('ssh://', '')
        if '/' in url:
            url = '/'.join(url.split('/')[1:])
    elif '@' in url and ':' in url:
        # git@github.com:user/repo.git -> user/repo.git
        url = url.split(':')[1]
    elif url.startswith('http://') or url.startswith('https://'):
        # https://github.com/user/repo.git -> user/repo.git
        url = '/'.join(url.split('/')[3:])

    # Extract just the repo name from the path
    if '/' in url:
        repo_name = url.split('/')[-1]
    else:
        repo_name = url

    # Remove .git extension if present
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]

    if not repo_name:
        return None

    return repo_name

def get_machine_info_with_interactive_fallback(machine_info: Path|None = None) -> Dict[str, Any]:
    machine_info_data = None
    if machine_info:
        with open(machine_info, 'r') as f:
            machine_info_data = json.load(f)
    else:
        format_terminal_output(GET_MACHINE_INFO_SCRIPT.read_text(), syntax="bash")
        print("No machine info file provided. Please paste the machine info JSON below. (e.g. by pasting the script printed above on the target machine)", file=sys.stderr)
        print("End with two blank lines:", file=sys.stderr)

        try:
            machine_info_text = read_until_double_newline()
            print("="*80 + "\nInput read", file=sys.stderr)
            machine_info_data = json.loads(machine_info_text)
        except (json.JSONDecodeError) as e:
            print(f"Error parsing machine info: {e}")
            sys.exit(1)

        if not machine_info_data:
            raise ValueError("Invalid machine info data")

    error = validate_machine_info(machine_info_data)
    if error:
        raise error

    return machine_info_data

def get_local_machine_info() -> Dict[str, Any]:
    """
    Execute the GET_MACHINE_INFO_SCRIPT locally and return the machine info.

    Returns:
        Dict containing the machine info from the local system

    Raises:
        RuntimeError: If the script fails, output is invalid JSON, or validation fails
    """
    try:
        result = subprocess.run(
            [str(GET_MACHINE_INFO_SCRIPT)],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute machine info script: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(f"Machine info script not found at: {GET_MACHINE_INFO_SCRIPT}")

    # Parse the JSON output
    try:
        machine_info_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse machine info script output as JSON: {e}")

    # Validate the machine info
    error = validate_machine_info(machine_info_data)
    if error:
        raise error

    return machine_info_data

def parse_args() -> argparse.Namespace:
    def str_to_bool_or_none(v):
        if v is None:
            return None
        if v.lower() == 'yes':
            return True
        if v.lower() == 'no':
            return False
        if v.lower() == 'auto':
            return None
        raise argparse.ArgumentTypeError(f"Invalid value: {v}. Must be 'yes', 'no', or 'auto'")

    p = argparse.ArgumentParser(
        description=textwrap.dedent("""
            Use this script to emit config scripts for system setup.
            The initial input is either a full config file
            (in which case the config is updated if any of the specifiying args are given),
            or a machine info file. (as obtained by running the contents of 'get-initial-machine-info.sh' on a target machine)
            (in which case the script will try to build a matching config, together with the other provided args)
        """)
    )
    p.add_argument("--config", "-c", type=Path, help="json file for a system config, e.g. as dumped with --dump-config")
    p.add_argument("--machine-info", "-mi", type=Path, help="The machine info json as obtained through the get-initial-machine-info.sh script on the target machine")
    p.add_argument("--dump-config", "-dc", action="store_true", help="Dump the effective configuration to stdout")
    p.add_argument("--target", "-t",
                   choices=[t.name.lower() for t in ExportTarget],
                   help="Export target")
    p.add_argument("--profile", "-p",
                   choices=[p.name.lower() for p in SetupProfile],
                   default="full_setup",
                   help="(default: full_setup) wether to generate A: a script for automated full setup or B: universal first stage (hardware specific setup on the target it self) or C: the second stage (for generation on the target it self, see --for-this-machine)")
    p.add_argument("--source-url", type=str, help="URL of the source repository (or automatically use the upstream of the PWD repo)")
    p.add_argument("--source-ref", type=str, help="Reference of the source repository (or automatically use the currently locally checked out branch)")
    p.add_argument("--sources-target-dir", type=str, help="The target directory in which to clone the repo into on the target")
    p.add_argument("--out", "-o", type=Path, help="Output file or directory (for dir_with_runner)")
    p.add_argument("--commands", "-cmd", type=str, nargs="*", help="The commands to run after the setup is complete. (as the given user, in the cloned repo dir)")
    p.add_argument("--for-this-machine", action="store_true", help="Generate configuration for this machine (runs machine info script locally)")
    p.add_argument("--clone-key", "-ck", type=Path, help="The private key file which allows cloning the private repo. (needs to not be password protected for autometic pull)")
    p.add_argument("--authorized-key-file", type=Path, help="Public key file to add to authorized keys")
    p.add_argument("--authorized-keys", type=str, nargs="*", help="Literal SSH public keys to add to authorized keys")
    p.add_argument("--forward-authorized-keys", "-fk", action="store_true", help="Also add all keys from local ~/.ssh/authorized_keys")
    p.add_argument("--local-private-key", "-k", type=Path, help="Local private key file to use (will create if doesn't exist)")
    p.add_argument("--user", type=str, help="Use the specified user, create if necessary (otherwise use the user from the provided system info)")
    p.add_argument("--management-user", type=str, help="For cloud-congig target: The sudo user to be created alongside the service user")
    p.add_argument("--gpu-vendor", choices=[v.vendor_name for v in GPU_VENDOR], help="The GPU vendor to use for the driver install - by default the provided machine info is used to try to determine the correct driver")
    p.add_argument("--install-drivers", type=str_to_bool_or_none, default=None, help="Whether to install accelerator/GPU drivers (yes/no/auto, default: auto) - auto relies on the provided machne info to detect if a driver is already installed")
    parsed_args = p.parse_args()

    # Check that either dump-config or target is specified
    if not parsed_args.dump_config and not parsed_args.target:
        print("Error: Either --dump-config or --target must be specified", file=sys.stderr)
        p.print_help()
        sys.exit(1)

    if parsed_args.for_this_machine and parsed_args.machine_info:
        print("Error: --for-this-machine and --machine-info specified at the same time, this makes no sense as --for-this-machine get the machine infor for this machine automatically", file=sys.stderr)
        p.print_help()
        sys.exit(1)

    return parsed_args

def load_authorized_keys(args: argparse.Namespace) -> list[str]:
    """Load authorized keys from various sources based on command line arguments."""
    keys = []

    # Load key from specified public key file
    if args.authorized_key_file and args.authorized_key_file.exists():
        try:
            key_content = args.authorized_key_file.read_text().strip()
            if key_content:
                keys.append(key_content)
        except Exception as e:
            print(f"Warning: Failed to read authorized key file {args.authorized_key_file}: {e}", file=sys.stderr)

    # Forward local authorized keys
    if args.forward_authorized_keys:
        print("Forwarding local authorized keys...")
        local_auth_keys = Path.home() / ".ssh" / "authorized_keys"
        if local_auth_keys.exists():
            try:
                with open(local_auth_keys, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            keys.append(line)
            except Exception as e:
                print(f"Warning: Failed to read local authorized_keys: {e}", file=sys.stderr)

    # Handle local private key (create if needed and add public key)
    if args.local_private_key:
        create_ssh_key_if_needed(args.local_private_key)
        pub_key_path = Path(str(args.local_private_key) + ".pub")
        if pub_key_path.exists():
            try:
                key_content = pub_key_path.read_text().strip()
                if key_content:
                    keys.append(key_content)
            except Exception as e:
                print(f"Warning: Failed to read public key {pub_key_path}: {e}", file=sys.stderr)

    # Handle direct list of authorized keys from command line
    if hasattr(args, 'authorized_keys') and args.authorized_keys:
        for key in args.authorized_keys:
            key = key.strip()
            if key and not key.startswith('#'):
                keys.append(key)

    return keys

def report_validation_errors_and_exit(errors: List[str], validation_type: str, pre_report_callable, permissive: bool):
    if not errors:
        return

    print(f"An error occurred while validating {validation_type}:", file=sys.stderr)

    pre_report_callable()

    print("\nErrors:\n" + "="*80)

    for i, e in enumerate(errors):
        print(f"Error {i+1}:\n" + _indent_block(e, 4), file=sys.stderr)

    if not permissive:
        exit(1)

def main() -> None:
    args = parse_args()

    cfg = None
    machine_info = None

    if args.for_this_machine:
        machine_info = get_local_machine_info()
    elif args.machine_info or not args.config:
        machine_info = get_machine_info_with_interactive_fallback(args.machine_info)

    gpu_vendor = next((v for v in GPU_VENDOR if v.vendor_name == args.gpu_vendor), None) if args.gpu_vendor else None

    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            cfg = Config.from_dict(config_data)

        # Select source_url in order: args -> config -> upstream
        if args.source_url:
            source_url = args.source_url
        elif cfg.source_url:
            source_url = cfg.source_url
        else:
            upstream_url, _ = get_upstream_repo_url_and_ref_for_pwd(
                private_key_path=args.clone_key
            )
            source_url = upstream_url

        # Select source_ref in order: args -> config -> upstream
        if args.source_ref:
            source_ref = args.source_ref
        elif cfg.source_ref:
            source_ref = cfg.source_ref
        else:
            # Always get from upstream if otherwise would be None
            _, upstream_ref = get_upstream_repo_url_and_ref_for_pwd(
                private_key_path=args.clone_key
            )
            source_ref = upstream_ref

        cfg = update_config(
            cfg=cfg,
            clone_key_path=args.clone_key,
            node_login_authorized_keys=load_authorized_keys(args),
            source_url=source_url,
            source_ref=source_ref,
            commands=args.commands,
            sources_target_dir=args.sources_target_dir if args.sources_target_dir else get_repo_name_from_git_url(cfg.source_url),
            gpu_vendor=gpu_vendor,
            user_override=args.user,
            install_drivers=args.install_drivers,
            management_user=args.management_user
        )
    elif machine_info:
        report_validation_errors_and_exit(
            machine_info_errors(machine_info),
            "the user provided machine info",
            lambda: format_terminal_output(json.dumps(machine_info, indent=2), syntax="json"),
            permissive=bool(args.dump_config)
        )
        source_url, source_ref = get_upstream_repo_url_and_ref_for_pwd(
            private_key_path=args.clone_key
        ) if not args.source_url else (args.source_url, None)

        if args.source_ref:
            source_ref = args.source_ref

        cfg = load_config(
            machine_info=machine_info,
            clone_key_path=args.clone_key,
            node_login_authorized_keys=load_authorized_keys(args),
            source_url=source_url,
            source_ref=source_ref,
            commands=args.commands,
            sources_target_dir=args.sources_target_dir if args.sources_target_dir else get_repo_name_from_git_url(source_url),
            gpu_vendor=gpu_vendor,
            user_override=args.user,
            install_drivers=args.install_drivers,
            management_user=args.management_user
        )
    else:
        raise RuntimeError("BUG: unreachable code")

    if machine_info:
        report_validation_errors_and_exit(
            config_against_machine_info_errors(cfg, machine_info),
            "the configuration against machine info",
            lambda: (
                format_terminal_output(json.dumps(cfg.to_dict(), indent=2), syntax="json", file=sys.stdout),
                format_terminal_output(json.dumps(machine_info, indent=2), syntax="json", file=sys.stdout)
            ),
            permissive=bool(args.dump_config)
        )

    report_validation_errors_and_exit(
        config_errors(cfg),
        "the generated configuration",
        lambda: format_terminal_output(json.dumps(cfg.to_dict(), indent=2), syntax="json", file=sys.stdout),
        permissive=bool(args.dump_config)
    )

    mods = [x() for x in allSetupModuleClasses]

    print_matrix_table(
        row_items=mods,
        col_items=list(SetupProfile)+[None]+list(ExportTarget),
        check_func=lambda mod, tgt_or_profile: (
            "-" if not mod.enabled(cfg) else (
                "P" if (tgt_or_profile in mod.profiles()) else "." if isinstance(tgt_or_profile, SetupProfile) else
                tgt_or_profile in mod.targets() if isinstance(tgt_or_profile, ExportTarget) else
                False
            )
        ),
        row_label="Module ↓ | Profiles/Targets →"
    )

    if args.dump_config:
        save_or_print_formatted(json.dumps(cfg.to_dict(), indent=2), syntax="json", path=args.out)
        sys.exit(0)

    target = ExportTarget[args.target.upper()]
    profile = SetupProfile[args.profile.upper()]

    active = [m for m in mods if (target in set(m.targets())) and (profile in set(m.profiles())) and m.enabled(cfg)]

    try:
        match target:
            case ExportTarget.CLOUD_INIT:
                save_or_print_formatted(emit_cloud_init(cfg, active), syntax="yaml", path=args.out)
            case ExportTarget.BASH_SCRIPT:
                save_or_print_formatted(emit_bash_script(cfg, active), syntax="bash", path=args.out)
            case ExportTarget.DIR_WITH_RUNNER:
                if not args.out:
                    print("--out required for dir_with_runner", file=sys.stderr)
                    sys.exit(2)
                emit_dir_with_runner(cfg, active, args.out)
                print(f"Wrote directory: {args.out}")
            case ExportTarget.SKY_LAUNCH:
                save_or_print_formatted(emit_skypilot_launch(cfg, active), syntax='yaml', path=args.out)
            case _:
                print("Unknown target", file=sys.stderr)
                sys.exit(2)
    except TemplateRenderError as ex:
        print("Error rendering template file:", file=sys.stderr)
        print("Context variables provided:", file=sys.stderr)
        format_terminal_output(json.dumps(ex.context, indent=2), syntax="json", file=sys.stderr)
        e = str(ex)
        print(f"\nError in template '{ex.template_file}': \n" + "="*len(e) + "\n" + e + "\n" + "="*len(e) + "\n", file=sys.stderr)
        exit(1)

# ------------ General helper functions ------------

def print_matrix_table(row_items, col_items, check_func, row_label="Item", file=sys.stderr):
    """
    Print a matrix table showing relationships between rows and columns.

    Args:
        row_items: List of items for rows
        col_items: List of items for columns (pass None for a separator column)
        check_func: Function(row_item, col_item) -> bool to check if cell should be marked
        row_label: Label for the row header
        file: File to print to (default stderr)
    """
    # Calculate the maximum width needed for row labels
    max_row_width = len(row_label)
    for row_item in row_items:
        row_name = row_item.name if hasattr(row_item, 'name') else str(row_item)
        max_row_width = max(max_row_width, len(row_name))

    # Add 2 spaces for padding
    label_column_width = max_row_width + 2

    # Print title
    print("-" * 80, file=file)

    # Print column names in a stairstep pattern
    for i, col in enumerate(col_items):
        if col is None:
            # Just add extra space for None columns
            print(" " * label_column_width + "│ " * i + " ", file=file)
        else:
            col_name = col.name if hasattr(col, 'name') else str(col)
            print(" " * label_column_width + "│ " * i + col_name, file=file)

    # Create header with connecting lines
    header = row_label.rjust(max_row_width) + "  "
    for i in range(len(col_items)):
        header += "│ "
    print(header, file=file)
    print("-" * 80, file=file)

    # Print each row
    for row_item in row_items:
        row_name = row_item.name if hasattr(row_item, 'name') else str(row_item)
        row = row_name[:max_row_width].rjust(max_row_width) + "  "
        for col_item in col_items:
            if col_item is None:
                row += "  "
                continue
            x = check_func(row_item, col_item)
            if x == True:
                row += "X "
            elif x == False:
                row += ". "
            else:
                row += x + " "

        print(row, file=file)

    print("-" * 80, file=file)

# ------------ Helper Functions invoking os commands ------------

def format_terminal_output(text: str, syntax: str = "yaml", file=sys.stdout) -> None:
    """Format terminal output with syntax highlighting if available."""

    # Check if we should use color
    term = os.environ.get("TERM", "")
    if not term or term == "dumb" or not sys.stdout.isatty():
        print(text, file=file)
        return

    # Check if highlight command exists
    if not shutil.which("highlight"):
        print(text, file=file)
        return

    try:
        # Use highlight to colorize the output
        result = subprocess.run(
            ["highlight", "-S", syntax, "-O", "xterm256"],
            input=text,
            text=True,
            capture_output=True,
            check=True
        )
        print(result.stdout, file=file)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to plain text if highlight fails
        print(text, file=file)

def save_or_print_formatted(text: str, syntax: str, path: Path|None):
    if not path:
        format_terminal_output(text, syntax=syntax)
        return

    # Map syntax to valid extensions (including alternate forms)
    syntax_to_valid_exts = {
        'bash': {'.sh'},
        'sh': {'.sh'},
        'yaml': {'.yaml', '.yml'},
        'yml': {'.yaml', '.yml'},
        'json': {'.json'},
    }

    # Check if path has a valid extension for the syntax
    valid_exts = syntax_to_valid_exts.get(syntax.lower(), set())

    if path.suffix == '':
        # No extension is fine, we'll save as-is
        pass
    elif valid_exts and path.suffix not in valid_exts:
        # Has extension but doesn't match the syntax
        print(f"Error: Path '{path}' has extension '{path.suffix}' which doesn't match syntax '{syntax}'", file=sys.stderr)
        print(f"Expected one of: {', '.join(sorted(valid_exts))} or no extension", file=sys.stderr)
        exit(1)

    try:
        path.write_text(text)
        print(f"Saved to: {path}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to save to {path}: {e}", file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    main()
