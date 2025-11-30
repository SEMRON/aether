from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any

from distqat.setup.common import *

@dataclass
class HostnameModule(Module):
    name: str = "set human readable hostname"

    def targets(self) -> Iterable[ExportTarget]:
        return [ExportTarget.CLOUD_INIT, ExportTarget.SKY_LAUNCH]

    def profiles(self) -> Iterable[SetupProfile]:
        return SetupProfile.without("STAGE_TWO_LOCAL_SETUP")

    def files(self, cfg: Config) -> Iterable[FileSpec]:
        return [FileSpec(Path("/usr/local/bin/generate-hostname.sh"), mode="0755", content=load_file(get_relative_path("generate-hostname.sh")))]

    def commands(self, cfg: Config) -> Iterable[str]:
        return ["/usr/local/bin/generate-hostname.sh"]
