from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

from distqat.setup.common import *

UNIT_NAME = "aether-final-commands.service"
UNIT_PATH = Path(f"/etc/systemd/system/{UNIT_NAME}")

@dataclass
class FinalRunnerModule(Module):
    name: str = "run the actual commands"

    def profiles(self) -> Iterable[SetupProfile]:
        return SetupProfile.without("STAGE_ONE_SETUP")

    def enabled(self, cfg: Config) -> bool:
        return bool(cfg.final_commands)

    def files(self, cfg: Config) -> Iterable[FileSpec]:
        script_content = "\n".join(["#!/bin/bash"] + cfg.final_commands)

        unit_content = f"""[Unit]
Description=Aether final commands (oneshot)
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User={cfg.username}
WorkingDirectory=/home/{cfg.username}/{cfg.sources_target_dir}
ExecStart=/usr/bin/bash /opt/aether/bin/final-commands.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""

        return [
            FileSpec(Path("/opt/aether/bin/final-commands.sh"), mode="0755", content=script_content),
            FileSpec(UNIT_PATH, mode="0644", content=unit_content),
        ]

    def commands(self, cfg: Config) -> Iterable[str]:
        return [
            "systemctl daemon-reload",
            f"systemctl start {UNIT_NAME}",
        ]
