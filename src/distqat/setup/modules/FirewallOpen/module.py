from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any

from distqat.setup.common import *

@dataclass
class FirewallOpenModule(Module):
    name: str = "open firewall"

    def enabled(self, cfg: Config) -> bool:
        return bool(cfg.open_ports)

    def targets(self) -> Iterable[ExportTarget]:
        return [x for x in list(ExportTarget) if not x in [ExportTarget.SKY_LAUNCH]]

    def commands(self, cfg: Config) -> Iterable[str]:
        cmds: List[str] = []

        if cfg.os.id == Distro.UBUNTU:
            cmds.append("if ! command -v ufw &> /dev/null; then echo 'Installing ufw...'; apt-get update && apt-get install -y ufw; fi")
            # because running the following commands might start the firewall, even if not previously running
            cmds.append("ufw allow ssh")
            for spec in cfg.open_ports:
                # spec like "100:65535/tcp" or "100/udp"
                if "/" in spec:
                    rng, proto = spec.split("/", 1)
                    cmds.append(f"ufw allow {rng.replace('-', ':')}/{proto}")
            cmds.append("ufw --force enable")
        elif cfg.os.platform_id in [
            Platform.FEDORA_39,
            Platform.FEDORA_40,
            Platform.FEDORA_41,
            Platform.FEDORA_42,
            Platform.ENTERPRISE_LINUX_8,
            Platform.ENTERPRISE_LINUX_9,
            Platform.AMAZON_LINUX_2022,
            Platform.AMAZON_LINUX_2023
        ]:
            cmds.append("if ! command -v firewall-cmd &> /dev/null; then echo 'Installing firewalld...'; dnf install -y firewalld; fi")
            cmds.append("systemctl enable --now firewalld")
            for spec in cfg.open_ports:
                # spec like "100-65535/tcp" or "100/udp"
                if "/" in spec:
                    rng, proto = spec.split("/", 1)
                    cmds.append(f"firewall-cmd --permanent --add-port={rng.replace(':', '-')}/{proto}")
            cmds.append("firewall-cmd --reload")
        else:
            raise RuntimeError(f"Unsupported OS for module {self.name}: {cfg.os}")

        return cmds
