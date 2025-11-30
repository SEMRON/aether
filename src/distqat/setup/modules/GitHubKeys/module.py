from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Dict, Any

from distqat.setup.common import *

@dataclass
class GitHubKeysModule(Module):
    name: str = "setup github known hosts"

    def files(self, cfg: Config) -> Iterable[FileSpec]:
        return [
            FileSpec(Path("/usr/local/bin/setup-github-known-hosts.sh"), mode="0755", content=load_file(get_relative_path("setup-github-known-hosts.sh"))),
        ]

    def commands(self, cfg: Config) -> Iterable[str]:
        return [
            "/usr/local/bin/setup-github-known-hosts.sh",
        ]
