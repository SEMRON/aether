"""
This file contains code for catching unsupported configurations.
Be that paltforms which are not supported,
or configurations which can not be supported for one reason or the other.
"""

import stat
import tempfile
import subprocess

from distqat.setup.common import *

def machine_info_errors(mi: Dict[str, Any]):
    errors = []

    if mi["cpu"]["architecture"] != "x86_64":
        errors.append("Only x86_64 architecture is supported.")

    return errors

def config_errors(cfg: Config):
    errors = []

    if not cfg.username:
        errors.append("Username (user to install distqat for) is not specified.")

    if not cfg.os.id and not cfg.os.platform_id:
        errors.append(f"Need match for either the OS or the platform to generate automatic installation script: id='{cfg.os.id}', platform_id='{cfg.os.platform_id}'")

    if cfg.install_drivers and not cfg.gpu_vendor:
        errors.append("GPU vendor is not specified, but requested to install drivers.")

    # If private key provided, test access
    if cfg.source_url:
        if cfg.deploy_key_private:
            # Create temporary file for private key
            private_key_fd, private_key_path = tempfile.mkstemp(suffix='.pem')
            private_key_path = Path(private_key_path)
            base_error_message = "Cannot validate deploy key: "

            try:
                # Write private key to temp file
                with os.fdopen(private_key_fd, 'w') as f:
                    f.write(cfg.deploy_key_private)

                # Set restrictive permissions (read/write for owner only)
                os.chmod(private_key_path, stat.S_IRUSR | stat.S_IWUSR)


                # Set up GIT_SSH_COMMAND to use the private key
                env = os.environ.copy()
                env["GIT_SSH_COMMAND"] = f"ssh -i {private_key_path.absolute()} -o StrictHostKeyChecking=no -o PasswordAuthentication=no -o PubkeyAuthentication=yes -o IdentitiesOnly=yes"

                # Test with git ls-remote (doesn't clone, just checks access)
                subprocess.run(
                    ["git", "ls-remote", cfg.source_url, "HEAD"],
                    capture_output=True,
                    check=True,
                    text=True,
                    env=env,
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                errors.append(base_error_message + f"Timeout while testing access to {cfg.source_url} with private key {private_key_path}")
            except subprocess.CalledProcessError as e:
                errors.append(base_error_message + f"Cannot access repository {cfg.source_url} with private key {private_key_path} - 'git ls-remote' stderr:\n{e.stderr}")
        else:
            # No private key provided, check if repo can be accessed without authorization
            base_error_message = "Cannot validate repository access: "

            try:
                # Test with git ls-remote without any authentication
                # Explicitly disable SSH keys to ensure no authentication is used
                env = os.environ.copy()
                env["GIT_SSH_COMMAND"] = "ssh -o PasswordAuthentication=no -o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"

                subprocess.run(
                    ["git", "ls-remote", cfg.source_url, "HEAD"],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=30,
                    env=env
                )
            except subprocess.TimeoutExpired:
                errors.append(base_error_message + f"Timeout while testing access to {cfg.source_url} without authentication")
            except subprocess.CalledProcessError as e:
                errors.append(base_error_message + f"Cannot access repository {cfg.source_url} without authentication - 'git ls-remote' stderr:\n{e.stderr}")

    # List of users/groups which likely already exist on systems, and might lead to failures
    # Especially because e.g. the admin user might not exist, but a group of the same name, which will lead to a collision
    reserved_user_names = [
        "daemon", "bin", "sys", "sync", "games", "man", "lp", "mail", "news", "uucp", "proxy", "www-data", "backup", "list", "irc", "gnats",
        "nobody", "systemd-network", "systemd-resolve", "syslog", "messagebus", "_apt", "lxd", "uuidd", "dnsmasq", "radvd", "sshd", "ftp", "postfix",
        "mysql", "postgres", "ntp", "chrony", "operator", "staff", "wheel", "sudo", "adm", "docker", "render", "video", "audio", "input", "tty",
        "admin", "centos", "debian", "fedora", "core",
    ]

    if cfg.management_user and cfg.management_user in reserved_user_names:
        errors.append(f"Management user '{cfg.management_user}' is reserved and cannot be used.")

    if cfg.username and cfg.username in reserved_user_names:
        errors.append(f"User '{cfg.username}' is reserved and cannot be used.")

    reserved_target_directories = [
        ".ssh", ".config", ".bashrc", ".bash_profile", ".profile", ".zshrc", ".vimrc", ".gitconfig",
        "bin", "sbin", "etc", "usr", "var", "tmp", "opt", "proc", "sys", "dev", "run", "boot", "lib", "lib64",
        "home", "mnt", "media", "srv", "snap", "swapfile", "lost+found",
        ".cache", ".local", ".npm", ".cargo", ".rustup", ".docker", ".kube", ".ansible",
        "Desktop", "Documents", "Downloads", "Music", "Pictures", "Videos", "Public", "Templates",
    ]

    if cfg.sources_target_dir:
        target_dir = Path(cfg.sources_target_dir)
        for part in target_dir.parts:
            if part in reserved_target_directories:
                errors.append(f"Sources target directory '{cfg.sources_target_dir}' contains reserved directory name '{part}'.")

    return errors

def config_against_machine_info_errors(cfg: Config, mi: Dict[str, Any]):
    errors = []

    if cfg.create_new_user and cfg.username == mi["user_info"]["username"]:
        errors.append(f"Username '{cfg.username}' already exists, but create_new_user is set to True.")

    if cfg.install_drivers and (mi["secure_boot_enabled"] != "false"):
        errors.append(
            "Secure Boot is enabled on the machine, but install_drivers is set to True.\n"
            "This is not supported because our driver installation needs DKMS, and we currently have no way to sign the driver."
        )

    if cfg.open_ports and not mi["network"]["has_public_ip"]:
        errors.append(
            "The machine does not have a public IP address, but is supposed to open ports.\n"
            "This makes no sense, because the machine will not be reachabe either way."
        )

    return errors
