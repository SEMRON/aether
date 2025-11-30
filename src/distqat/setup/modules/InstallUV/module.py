from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any

from distqat.setup.common import *

@dataclass
class InstallUVModule(Module):
    name: str = "install uv"

    def profiles(self) -> Iterable[SetupProfile]:
        return SetupProfile.without("STAGE_ONE_SETUP")

    def commands(self, cfg: Config) -> Iterable[str]:
        return ["curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh"]
