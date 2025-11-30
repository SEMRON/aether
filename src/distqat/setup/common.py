from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, List, Optional, Protocol, Dict, Any
from pathlib import Path
import subprocess
from jinja2 import Template, StrictUndefined
import sys
import os
import inspect

# ------------ Core types ------------

class ExportTarget(Enum):
    CLOUD_INIT = auto()
    BASH_SCRIPT = auto()
    DIR_WITH_RUNNER = auto()
    SKY_LAUNCH = auto()  # SkyPilot launch.yaml

class SetupProfile(Enum):
    STAGE_ONE_SETUP = auto()
    STAGE_TWO_LOCAL_SETUP = auto()
    FULL_SETUP = auto()

    @classmethod
    def without(cls, *excluded_profiles):
        """Returns a list of all SetupProfiles except those provided."""
        # Convert any string exclusions to SetupProfile enum members
        processed_exclusions = []
        for profile in excluded_profiles:
            if isinstance(profile, str):
                # Try to find matching enum member (case insensitive)
                found = False
                profile_upper = profile.upper()
                for member in cls:
                    # Try exact match first (case insensitive), then with _SETUP suffix
                    if member.name.upper() in [profile_upper, profile_upper + "_SETUP"]:
                        processed_exclusions.append(member)
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"No SetupProfile found matching string '{profile}'")
            else:
                processed_exclusions.append(profile)

        return [profile for profile in cls if profile not in processed_exclusions]

class Distro(Enum):
    UBUNTU = "ubuntu"
    FEDORA = "fedora"
    DEBIAN = "debian"
    RHEL = "rhel"
    CENTOS = "centos"
    AMAZON = "amzn"

class Platform(Enum):
    FEDORA_39 = "platform:f39"
    FEDORA_40 = "platform:f40"
    FEDORA_41 = "platform:f41"
    FEDORA_42 = "platform:f42"
    ENTERPRISE_LINUX_8 = "platform:el8"
    ENTERPRISE_LINUX_9 = "platform:el9"
    AMAZON_LINUX_2022 = "platform:al2022"
    AMAZON_LINUX_2023 = "platform:al2023"

@dataclass
class OS:
    id: Distro|None
    version_id: str|None
    platform_id: Platform|None

class GPU_VENDOR(Enum):
    NO = ("no_gpu", None)
    NVIDIA = ("nvidia", "10de")
    AMD = ("amd", "1002")

    def __init__(self, name: str, pci_vendor_id: str|None):
        self.vendor_name = name
        self.pci_vendor_id = pci_vendor_id

    def __str__(self) -> str:
        return self.vendor_name

@dataclass
class Config:
    os: OS
    username: str
    management_user: Optional[str]
    create_new_user: bool
    gpu_vendor: Optional[GPU_VENDOR]
    source_url: Optional[str]
    source_ref: Optional[str]
    sources_target_dir: Optional[str]
    deploy_key_private: Optional[str]
    deploy_key_public: Optional[str]
    authorized_pubkey: List[str]
    final_commands: List[str]
    open_ports: List[str]
    install_drivers: Optional[bool]
    extras: Dict[str, Any] = field(default_factory=dict) # free-form

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "os": {
                "id": self.os.id.value if self.os.id else None,
                "version_id": self.os.version_id,
                "platform_id": self.os.platform_id.value if self.os.platform_id else None
            },
            "authorized_pubkey": self.authorized_pubkey,
            "create_new_user": self.create_new_user,
            "deploy_key_private": self.deploy_key_private,
            "deploy_key_public": self.deploy_key_public,
            "extras": self.extras,
            "gpu_vendor": str(self.gpu_vendor) if self.gpu_vendor else None,
            "install_drivers": self.install_drivers,
            "final_commands": self.final_commands,
            "open_ports": self.open_ports,
            "source_url": self.source_url,
            "source_ref": self.source_ref,
            "sources_target_dir": self.sources_target_dir,
            "username": self.username,
            "management_user": self.management_user,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Load Config from a dictionary (e.g., from JSON)."""
        os_data = data.get("os", {})
        os_obj = OS(
            id=Distro(os_data["id"]) if os_data.get("id") else None,
            version_id=os_data.get("version_id"),
            platform_id=Platform(os_data["platform_id"]) if os_data.get("platform_id") else None
        )

        gpu_vendor = None
        if data.get("gpu_vendor"):
            for vendor in GPU_VENDOR:
                if vendor.vendor_name == data["gpu_vendor"]:
                    gpu_vendor = vendor
                    break

        return cls(
            os=os_obj,
            username=data["username"],
            management_user=data.get("management_user"),
            create_new_user=data.get("create_new_user", False),
            gpu_vendor=gpu_vendor,
            source_url=data.get("source_url"),
            source_ref=data.get("source_ref"),
            sources_target_dir=data.get("sources_target_dir"),
            deploy_key_private=data.get("deploy_key_private"),
            deploy_key_public=data.get("deploy_key_public"),
            authorized_pubkey=data.get("authorized_pubkey", []),
            open_ports=data.get("open_ports", []),
            final_commands=data.get("final_commands", []),
            install_drivers=data.get("install_drivers"),
            extras=data.get("extras", {})
        )

@dataclass
class FileSpec:
    path: Path
    mode: str = "0644"
    user: str = "root"
    group: str = "root"
    content: str = ""

    def owner(self):
        return f"{self.user}:{self.group}"

class Module(Protocol):
    name: str

    def enabled(self, cfg: Config) -> bool:
        return True

    def targets(self) -> Iterable[ExportTarget]:
        return list(ExportTarget)

    def profiles(self) -> Iterable[SetupProfile]:
        return list(SetupProfile)

    def files(self, cfg: Config) -> Iterable[FileSpec]:
        return []

    def commands(self, cfg: Config) -> Iterable[str]:
        return []

# -- helpers --

class TemplateRenderError(Exception):
    """Error raised when template rendering fails."""
    def __init__(self, message: str, template_file: str = None, context: Dict[str, Any] = None):
        self.template_file = template_file
        self.context = context
        super().__init__(message)

def load_file(path: str, cfg: Dict[str, Any]|None = None) -> str:
    """Load content of a file relative to this script's directory, optionally parametrizing with Jinja2."""
    script_dir = Path(__file__).parent
    file_path = script_dir / path

    def raise_error(e: str):
        raise TemplateRenderError("Error emitted from within jinja template" + e, path, cfg)

    try:
        content = file_path.read_text()

        # If cfg is provided, use Jinja2 to parametrize the content
        if cfg is not None:
            try:
                template = Template(content, undefined=StrictUndefined)
                content = template.render(**cfg, raise_error=raise_error)
            except Exception as e:
                raise TemplateRenderError(f"Jinja error: {e}", path, cfg) from e

        return content
    except TemplateRenderError as e:
        raise e
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {e}")

def get_relative_path(relative_path):
    """Get the absolute path of a file relative to the calling file's directory."""
    # Get the frame of the caller
    caller_frame = inspect.stack()[1]
    # Get the filename of the caller
    caller_file = caller_frame.filename
    # Get the directory of the caller file
    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    # Construct and return the absolute path
    return os.path.join(caller_dir, relative_path)

# ------------ Helper Functions invoking os commands ------------

def create_ssh_key_if_needed(key_path: Path) -> None:
    """Create an SSH key pair if the private key doesn't exist."""

    if not key_path.exists():
        key_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ssh-keygen",
            "-t", "ed25519",
            "-f", str(key_path),
            "-N", "",  # No passphrase
            "-C", f"aether@{os.uname().nodename}"
        ], check=True)
        print(f"Created new SSH key pair: {key_path}", file=sys.stderr)

    # Also ensure the public key exists
    pub_key_path = Path(str(key_path) + ".pub")
    if not pub_key_path.exists():
        # Generate public key from private key
        subprocess.run([
            "ssh-keygen",
            "-y",
            "-f", str(key_path)
        ], stdout=open(pub_key_path, 'w'), check=True)
        print(f"Generated public key from private key: {pub_key_path}", file=sys.stderr)
