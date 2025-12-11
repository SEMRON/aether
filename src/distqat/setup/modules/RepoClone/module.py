from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any

from distqat.setup.common import *

@dataclass
class RepoCloneModule(Module):
    name: str = "clone repo"

    def profiles(self) -> Iterable[SetupProfile]:
        return SetupProfile.without("STAGE_TWO_LOCAL_SETUP")

    def files(self, cfg: Config) -> Iterable[FileSpec]:
        return [
            FileSpec(Path("/opt/aether/repo_url"), mode="0755", content=str(cfg.source_url)),
            FileSpec(Path("/opt/aether/bin/clone-repo.sh"), mode="0755", content=load_file(get_relative_path("clone-repo.sh.jinja2"), cfg=cfg.to_dict())),
        ]

    def commands(self, cfg: Config) -> Iterable[str]:
        cmds = []
        if cfg.os.id in [Distro.UBUNTU, Distro.DEBIAN]:
            cmds.append("if ! command -v git &> /dev/null; then apt-get update && apt-get install -y git; fi")
        elif cfg.os.id in [Distro.FEDORA, Distro.RHEL, Distro.CENTOS, Distro.AMAZON] or \
             cfg.os.platform_id in [Platform.FEDORA_39, Platform.FEDORA_40, Platform.FEDORA_41, Platform.FEDORA_42,
                                    Platform.ENTERPRISE_LINUX_8, Platform.ENTERPRISE_LINUX_9,
                                    Platform.AMAZON_LINUX_2022, Platform.AMAZON_LINUX_2023]:
            cmds.append("if ! command -v git &> /dev/null; then dnf install -y git; fi")

        cmds.append(f"sudo -u {cfg.username} /opt/aether/bin/clone-repo.sh")
        return cmds
