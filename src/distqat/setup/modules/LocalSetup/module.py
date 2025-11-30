from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any
import json

from distqat.setup.common import *

@dataclass
class LocalSetupModule(Module):
    name: str = "second stage local auto setup"

    def profiles(self) -> Iterable[SetupProfile]:
        return [SetupProfile.STAGE_ONE_SETUP]

    def files(self, cfg: Config) -> Iterable[FileSpec]:
        return [
            FileSpec(Path("/opt/aether/config.json"), mode="0755", content=str(json.dumps(cfg.to_dict()))),
        ]

    def commands(self, cfg: Config) -> Iterable[str]:
        generator_script_path_relative_to_repo_root = "setup/create-setup-files.py"

        return [
            f"sudo -u {cfg.username} python3 ~{cfg.username}/{cfg.sources_target_dir}/{generator_script_path_relative_to_repo_root} -c /opt/aether/config.json -t bash_script -p stage_two_local_setup -o ~{cfg.username}/second-stage-setup-$$.sh",
            f"bash ~{cfg.username}/second-stage-setup-$$.sh",
        ]
