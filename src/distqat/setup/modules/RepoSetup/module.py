from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any

from distqat.setup.common import *

@dataclass
class RepoSetupModule(Module):
    name: str = "setup repo"

    def profiles(self) -> Iterable[SetupProfile]:
        return SetupProfile.without("STAGE_ONE_SETUP")

    def files(self, cfg: Config) -> Iterable[FileSpec]:
        return [
            FileSpec(Path("/opt/aether/bin/install-repo.sh"), mode="0755", content=load_file(get_relative_path("install-repo.sh.jinja2"), cfg=cfg.to_dict()))
        ]

    def commands(self, cfg: Config) -> Iterable[str]:
        return [
            f"sudo -u {cfg.username} /opt/aether/bin/install-repo.sh",
        ]
