from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any

from distqat.setup.common import *

@dataclass
class CreateUserModule(Module):
    name: str = "create user"

    def enabled(self, cfg: Config) -> bool:
        return cfg.create_new_user

    def profiles(self) -> Iterable[SetupProfile]:
        return SetupProfile.without("STAGE_TWO_LOCAL_SETUP")

    def targets(self):
        return [ExportTarget.BASH_SCRIPT, ExportTarget.DIR_WITH_RUNNER]

    def commands(self, cfg: Config) -> Iterable[str]:
        commands = [f"id {cfg.username} &>/dev/null || useradd -m {cfg.username}"]

        # Add authorized keys if configured
        if cfg.authorized_pubkey:
            commands.extend([
                f"mkdir -p /home/{cfg.username}/.ssh",
                f"chmod g-w,o-w /home/{cfg.username}/.ssh",
                f"touch /home/{cfg.username}/.ssh/authorized_keys",
                f"chmod 600 /home/{cfg.username}/.ssh/authorized_keys",
            ])

            # Add each public key
            for pubkey in cfg.authorized_pubkey:
                commands.append(
                    f"echo '{pubkey}' >> /home/{cfg.username}/.ssh/authorized_keys"
                )

            # Fix ownership
            commands.append(
                f"chown -R {cfg.username}:$(id -gn {cfg.username}) /home/{cfg.username}/.ssh"
            )

        return commands
