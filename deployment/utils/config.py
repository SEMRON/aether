from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


# Provider-specific deployment dataclass for libvirt
@dataclass
class LibvirtDeployment:
    image: str
    ram: str
    vcpus: int
    os_variant: str
    network: str
    extraopts: Optional[List[str]] = None


# Provider-specific deployment dataclass for Prime Intellect
@dataclass
class PrimeIntellectDeployment:
    gpu_type: str
    gpu_count: int
    team_id: str
    regions: List[str] = field(
        default_factory=lambda: ["eu_west", "eu_north", "eu_east", "united_states"]
    )
    os_prefix: str = "ubuntu_22"
    provider_blacklist: List[str] = field(default_factory=lambda: ["runpod"])
    disk_size: Optional[int] = None
    env_vars: Optional[Dict[str, str]] = None
    # Optional: explicit cloud ID to use (bypasses gpu_type search if specified)
    cloud_id: Optional[str] = None
    # Optional: timeout in seconds for waiting for pod to become active (default: 300)
    provision_timeout: int = 300


# Dispatcher for provider-specific deployment parsing
def parse_deploymentinfo(provider: str, config: Dict[str, Any]) -> Any:
    if provider == "libvirt":
        return LibvirtDeployment(
            image=config["image"],
            ram=str(config["ram"]),
            vcpus=int(config["vcpus"]),
            os_variant=config["os-variant"],
            network=config["network"],
            extraopts=config.get("extraopts"),
        )
    elif provider == "prime_intellect":
        return PrimeIntellectDeployment(
            gpu_type=config.get("gpu_type", ""),
            gpu_count=int(config.get("gpu_count", 1)),
            team_id=config["team_id"],
            regions=config.get(
                "regions", ["eu_west", "eu_north", "eu_east", "united_states"]
            ),
            os_prefix=config.get("os_prefix", "ubuntu_22"),
            provider_blacklist=config.get("provider_blacklist", ["runpod"]),
            disk_size=config.get("disk_size"),
            env_vars=config.get("env_vars"),
            cloud_id=config.get("cloud_id"),
            provision_timeout=int(config.get("provision_timeout", 300)),
        )
    return None


@dataclass
class Node:
    name: str
    type_name: str
    os: str
    address: Optional[str] = None
    deployment: Optional[Any] = None


def parse_nodes(config: Dict[str, Any]) -> Generator[Node, None, None]:
    nodes_cfg = config.get("nodes", {})
    for type_name, type_cfg in nodes_cfg.items():
        array_cfg = type_cfg.get("array")
        deployment_cfg = type_cfg.get("deployment")

        deployment = None
        if deployment_cfg:
            provider = deployment_cfg.get("provider")
            deployment_config = deployment_cfg.get("config", {})
            deployment = parse_deploymentinfo(provider, deployment_config)

        if array_cfg:
            count = array_cfg["count"]
            name_template = array_cfg.get(
                "name_template", f"{type_name}_{{array_name}}"
            )
            names = array_cfg.get("names", [str(i + 1) for i in range(count)])
            addresses = array_cfg.get("addresses", [None] * count)

            if len(names) != count:
                raise ValueError("Length of names list must match count")
            if addresses and len(addresses) != count:
                raise ValueError("Length of addresses list must match count")

            for i in range(count):
                name = name_template.format(array_name=names[i])
                node = Node(
                    name=name,
                    type_name=type_name,
                    os=type_cfg["os"],
                    address=addresses[i] if addresses[i] else None,
                    deployment=deployment,
                )
                yield node
        else:
            name = type_cfg.get("name", type_name)
            address = type_cfg.get("address")
            node = Node(
                name=name,
                type_name=type_name,
                os=type_cfg["os"],
                address=address,
                deployment=deployment,
            )
            yield node
