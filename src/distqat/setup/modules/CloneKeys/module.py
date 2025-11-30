from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any

from distqat.setup.common import *

@dataclass
class CloneKeysModule(Module):
    name: str = "configure git clone keys"

    def enabled(self, cfg: Config) -> bool:
        # Check if both keys are set or both are unset
        has_private = cfg.deploy_key_private is not None
        has_public = cfg.deploy_key_public is not None

        if has_private != has_public:
            raise ValueError("Both deploy_key_private and deploy_key_public must be set, or neither")

        return has_private and has_public

    def profiles(self) -> Iterable[SetupProfile]:
        return SetupProfile.without("STAGE_TWO_LOCAL_SETUP")

    def files(self, cfg: Config) -> Iterable[FileSpec]:
        files: List[FileSpec] = []
        if cfg.deploy_key_private:
            files.append(FileSpec(Path("/opt/aether/clone-keys/clone_key"), mode="400", user=cfg.username, group=cfg.username, content=cfg.deploy_key_private))
        if cfg.deploy_key_public:
            files.append(FileSpec(Path("/opt/aether/clone-keys/clone_key.pub"), mode="400", user=cfg.username, group=cfg.username, content=cfg.deploy_key_public))
        return files

    def commands(self, cfg: Config) -> Iterable[str]:
        cmds = [
            "mkdir -p /opt/aether/clone-keys",
            f"chown {cfg.username}:{cfg.username} /opt/aether/clone-keys",
            "chmod 750 /opt/aether/clone-keys",
            "echo \"-i /opt/aether/clone-keys/clone_key\" > /opt/aether/ssh_args.clone_key",
        ]
        return cmds
