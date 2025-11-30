import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import ipaddress

import torch
torch.multiprocessing.set_sharing_strategy('file_descriptor')

from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

from typing import Dict, List, Optional, Set
import time
import click
import wandb
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.dht import DHT
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import SchemaValidator
from hivemind.optim.progress_tracker import TrainingProgressSchema

from distqat.config import Config
from distqat.config import parse_args
from distqat.distributed.optim.diloco import TrainingState
from distqat.utils.metrics import LocalMetrics
from distqat.distributed.utils.auto_discovery import get_stage_name, discover_experts
from distqat.distributed.utils.networking import strip_port, get_port
from distqat.utils.logging import store_wandb_run_id, get_wandb_run_id
import secrets
import string

logger = get_logger(__name__)
use_hivemind_log_handler("in_root_logger")


@dataclass
class NodeInfo:
    peer_id: str
    last_contact: float
    stage_index: Optional[int] = None
    stage_name: Optional[str] = None
    address: Optional[str] = None
    multiaddrs: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)

    def last_contact_iso(self) -> str:
        return datetime.utcfromtimestamp(self.last_contact).isoformat()

    def display_identifier(self) -> str:
        return self.address or self.peer_id


@dataclass
class StageInfo:
    stage_index: int
    stage_name: str
    num_peers: int
    nodes: List[NodeInfo] = field(default_factory=list)
    missing: bool = False
    endpoint: Optional[str] = None


@dataclass
class PipelineInfo:
    pipeline_index: int
    stages: List[StageInfo] = field(default_factory=list)


@dataclass
class LogEntry:
    step: int
    timestamp: float
    distributed_loss: Optional[float] = None
    baseline_loss: Optional[float] = None
    pipelines: List[PipelineInfo] = field(default_factory=list)

    def as_console_message(self) -> str:
        distributed_loss_str = f"{self.distributed_loss:.5f}" if self.distributed_loss is not None else "n/a"
        baseline_loss_str = f"{self.baseline_loss:.5f}" if self.baseline_loss is not None else "n/a"

        lines = [
            f"[Step {self.step}]",
            "  Losses:",
            f"    distributed: {distributed_loss_str}",
            f"    baseline:    {baseline_loss_str}",
        ]

        lines.append("  Pipelines:")
        if self.pipelines:
            for pipeline in self.pipelines:
                lines.append(f"    Pipeline #{pipeline.pipeline_index}:")
                if pipeline.stages:
                    for stage in pipeline.stages:
                        stage_header = f"      - {stage.stage_name}: {stage.num_peers} peer(s)"
                        if stage.missing:
                            stage_header += " [missing]"
                        lines.append(stage_header)
                        if stage.nodes:
                            for node in stage.nodes:
                                display_name = node.display_identifier()
                                lines.append(f"          - {display_name}")
                                lines.append(f"              last_contact: {node.last_contact_iso()}")
                                if node.address and node.peer_id:
                                    lines.append(f"              peer_id: {node.peer_id}")
                                elif not node.address and node.peer_id != display_name:
                                    lines.append(f"              peer_id: {node.peer_id}")
                                resolved_endpoint = next(
                                    (
                                        endpoint
                                        for endpoint in node.endpoints
                                        if endpoint
                                    ),
                                    None,
                                )
                                if resolved_endpoint and resolved_endpoint != display_name:
                                    lines.append(f"              endpoint: {resolved_endpoint}")
                                elif node.multiaddrs:
                                    lines.append(f"              multiaddr: {node.multiaddrs[0]}")
                        else:
                            if stage.missing:
                                lines.append("          (stage missing)")
                            else:
                                lines.append("          (no active peers)")
                else:
                    lines.append("      (no stages)")
        else:
            lines.append("    (no pipeline data)")

        return "\n".join(lines)

    def to_wandb_payload(self):
        unique_peers = {
            node.peer_id
            for pipeline in self.pipelines
            for stage in pipeline.stages
            for node in stage.nodes
        }
        total_stages = sum(len(pipeline.stages) for pipeline in self.pipelines)

        payload = {
            "monitor/step": self.step,
            "monitor/timestamp": self.timestamp,
            "monitor/pipelines_count": len(self.pipelines),
            "monitor/stages_count": total_stages,
            "monitor/nodes_count": len(unique_peers),
        }

        if self.distributed_loss is not None:
            payload["loss/distributed"] = self.distributed_loss
        if self.baseline_loss is not None:
            payload["loss/baseline"] = self.baseline_loss

        for pipeline in self.pipelines:
            for stage in pipeline.stages:
                payload[f"monitor/pipeline_{pipeline.pipeline_index}/stage_{stage.stage_index}/peers"] = stage.num_peers
                payload[f"monitor/pipeline_{pipeline.pipeline_index}/stage_{stage.stage_index}/missing"] = int(stage.missing)

        if unique_peers:
            payload["monitor/nodes_table"] = wandb.Table(
                columns=["pipeline", "stage", "address", "peer_id", "last_contact"],
                data=[
                    [
                        pipeline.pipeline_index,
                        stage.stage_name if stage.stage_name is not None else "unknown",
                        (
                            node.address
                            or (node.endpoints[0] if node.endpoints else None)
                            or (node.multiaddrs[0] if node.multiaddrs else None)
                            or node.peer_id
                        ),
                        node.peer_id,
                        node.last_contact_iso(),
                    ]
                    for pipeline in self.pipelines
                    for stage in pipeline.stages
                    for node in stage.nodes
                ],
            )

        return payload


class Monitor:
    def __init__(
        self, 
        config: Config, 
        refresh_period: int = 300,
        store_ip_addresses_path: Optional[str] = None
    ):
        self.config = config
        self.refresh_period = refresh_period
        self.dht = DHT(
            start=True,
            initial_peers=config.network.initial_peers,
            host_maddrs=config.network.host_maddrs,
            announce_maddrs=config.network.announce_maddrs,
            identity_path=config.network.identity_path,
        )

        signature_validator = RSASignatureValidator()
        self.dht.add_validators([SchemaValidator(TrainingProgressSchema, prefix=config.experiment_prefix), signature_validator])

        # Initialize wandb only if wandb_project is set
        self.wandb_enabled = config.wandb_project is not None
        if self.wandb_enabled:
            wandb_run_id = config.wandb_run_id
            if wandb_run_id is None:
                wandb_run_id = get_wandb_run_id(self.dht, config.experiment_prefix)
                if wandb_run_id is None:
                    wandb_run_id = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
                    store_wandb_run_id(self.dht, config.experiment_prefix, wandb_run_id, expiration=3600)
                    logger.info(f"Generated new wandb_run_id: {wandb_run_id} and stored in DHT")
                else:
                    logger.info(f"Retrieved existing wandb_run_id from DHT: {wandb_run_id}")
            else:
                store_wandb_run_id(self.dht, config.experiment_prefix, wandb_run_id, expiration=3600)
                logger.info(f"Stored wandb_run_id in DHT: {wandb_run_id}")
            
            try:
                wandb.init(
                    entity=config.wandb_entity,
                    project=config.wandb_project,
                    name=config.experiment_prefix,
                    id=wandb_run_id,
                    settings=wandb.Settings(
                        x_label="monitor",
                    ),
                )
                logger.debug(f"Wandb initialized: {config.wandb_entity} {config.wandb_project} {config.experiment_prefix} {wandb_run_id}")
                wandb.config.update(config.model_dump())
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
                self.wandb_enabled = False
        else:
            logger.info("Wandb project not set, skipping wandb initialization")
    
        self.store_ip_addresses_path = store_ip_addresses_path

    @staticmethod
    async def _list_peer_multiaddrs(_dht, node) -> Dict[str, List[str]]:
        peers = await node.p2p.list_peers()
        return {
            peer.peer_id.to_bytes().hex(): [str(addr) for addr in peer.addrs]
            for peer in peers
        }

    def _get_peer_multiaddrs(self) -> Dict[str, List[str]]:
        try:
            return self.dht.run_coroutine(Monitor._list_peer_multiaddrs)
        except Exception as e:
            logger.debug(f"Failed to fetch peer multiaddrs: {e}")
            return {}

    @staticmethod
    def _validate_ip(candidate: Optional[str]) -> Optional[str]:
        if candidate is None:
            return None
        candidate = candidate.strip()
        if not candidate:
            return None
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            return None

    @classmethod
    def _extract_ip_from_multiaddr(cls, multiaddr: str) -> Optional[str]:
        if not multiaddr:
            return None
        if multiaddr.startswith("/"):
            parts = multiaddr.strip("/").split("/")
            for index in range(0, len(parts) - 1, 2):
                protocol = parts[index]
                value = parts[index + 1]
                if protocol in {"ip4", "ip6"}:
                    candidate = cls._validate_ip(value)
                    if candidate:
                        return candidate
        return None

    @classmethod
    def _extract_ip_from_endpoint(cls, endpoint: Optional[str]) -> Optional[str]:
        if not endpoint:
            return None
        if endpoint.startswith("/"):
            return cls._extract_ip_from_multiaddr(endpoint)
        host_part = strip_port(endpoint)
        host_part = host_part.strip("[]")
        candidate = cls._validate_ip(host_part)
        if candidate:
            return candidate
        return None

    @classmethod
    def _select_preferred_ip(cls, addresses: List[str]) -> Optional[str]:
        candidates = []
        for address in addresses:
            candidate = cls._extract_ip_from_multiaddr(address)
            if candidate:
                try:
                    candidates.append(ipaddress.ip_address(candidate))
                except ValueError:
                    continue
        for ip_obj in candidates:
            if ip_obj.is_global:
                return str(ip_obj)
        for ip_obj in candidates:
            if not (ip_obj.is_loopback or ip_obj.is_unspecified):
                return str(ip_obj)
        if candidates:
            return str(candidates[0])
        return None


    def _collect_progress_info(self) -> List[PipelineInfo]:
        stage_contact_map: Dict[int, Dict[str, float]] = defaultdict(dict)
        stage_name_map: Dict[int, str] = {}
        peer_multiaddrs = self._get_peer_multiaddrs()
        peer_ip_map = {
            peer_id: self._select_preferred_ip(addresses)
            for peer_id, addresses in peer_multiaddrs.items()
        }

        n_stages = len(self.config.model_pipeline.pipeline)
        experiment_prefix = self.config.experiment_prefix

        for stage_index in range(n_stages):
            stage_name = get_stage_name(self.config, stage_index)
            stage_name_map[stage_index] = stage_name
            progress_entries: List[TrainingState] = []
            progress_response = self.dht.get(f"{experiment_prefix}_{stage_index}_progress", latest=True)

            if progress_response is not None:
                progress_dict = progress_response.value
                if isinstance(progress_dict, dict):
                    progress_entries = [
                        TrainingState.validate(entry.value)
                        for entry in progress_dict.values()
                        if entry.value is not None
                    ]
                    for state in progress_entries:
                        peer_id = state.peer_id.hex()
                        stage_contact_map[stage_index][peer_id] = max(
                            stage_contact_map[stage_index].get(peer_id, float("-inf")), state.time
                        )

        stage_nodes_map: Dict[int, List[NodeInfo]] = {}
        for stage_index in range(n_stages):
            stage_nodes_map[stage_index] = [
                NodeInfo(
                    peer_id=peer_id,
                    last_contact=last_contact,
                    stage_index=stage_index,
                    stage_name=stage_name_map.get(stage_index),
                    address=peer_ip_map.get(peer_id),
                    multiaddrs=peer_multiaddrs.get(peer_id, []).copy(),
                )
                for peer_id, last_contact in sorted(
                    stage_contact_map[stage_index].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]

        # Discover pipeline structure from the DHT (best effort)
        pipeline_stage_metadata: Dict[int, Dict[str, Dict[str, str]]] = {}
        pipeline_missing_stage_map: Dict[int, Set[str]] = defaultdict(set)
        discovered_pipeline_indices: List[int] = []
        try:
            complete_pipelines, incomplete_pipelines = discover_experts(self.dht, self.config)
            for pipeline_index, stages in complete_pipelines.items():
                pipeline_stage_metadata[pipeline_index] = stages
            for pipeline_index, data in incomplete_pipelines.items():
                pipeline_stage_metadata.setdefault(pipeline_index, {}).update(data.get("stages", {}))
                pipeline_missing_stage_map[pipeline_index].update(data.get("missing_stages", []))
            discovered_pipeline_indices = sorted(
                set(pipeline_stage_metadata.keys()) | set(pipeline_missing_stage_map.keys())
            )
        except Exception as e:
            logger.debug(f"Pipeline discovery failed: {e}")

        pipeline_count_nodes = max((len(nodes) for nodes in stage_nodes_map.values()), default=0)
        pipeline_count = max(pipeline_count_nodes, len(discovered_pipeline_indices))

        if pipeline_count == 0:
            return []

        if discovered_pipeline_indices:
            pipeline_indices = list(discovered_pipeline_indices)
            if len(pipeline_indices) < pipeline_count:
                next_index = pipeline_indices[-1] + 1 if pipeline_indices else 0
                while len(pipeline_indices) < pipeline_count:
                    if next_index not in pipeline_indices:
                        pipeline_indices.append(next_index)
                    next_index += 1
        else:
            pipeline_indices = list(range(pipeline_count))

        pipeline_indices.sort()
        if len(pipeline_indices) > pipeline_count:
            pipeline_count = len(pipeline_indices)

        # Choose a reference stage (one with the most nodes) to align pipelines across stages.
        reference_stage_index = 0
        if pipeline_count_nodes > 0:
            for stage_index in range(n_stages):
                if len(stage_nodes_map[stage_index]) == pipeline_count_nodes:
                    reference_stage_index = stage_index
                    break

        stage_assignments: Dict[int, List[Optional[NodeInfo]]] = {
            stage_index: [None] * pipeline_count for stage_index in range(n_stages)
        }

        reference_nodes = stage_nodes_map.get(reference_stage_index, [])
        for ordinal in range(min(len(reference_nodes), pipeline_count)):
            stage_assignments[reference_stage_index][ordinal] = reference_nodes[ordinal]

        for stage_index in range(n_stages):
            if stage_index == reference_stage_index:
                continue
            nodes = stage_nodes_map.get(stage_index, [])
            if not nodes:
                continue

            remaining_nodes = nodes.copy()
            if pipeline_count_nodes == 0 or not reference_nodes:
                for ordinal in range(min(len(remaining_nodes), pipeline_count)):
                    stage_assignments[stage_index][ordinal] = remaining_nodes[ordinal]
                continue

            for ordinal in range(pipeline_count):
                if not remaining_nodes:
                    break
                reference_node = stage_assignments[reference_stage_index][ordinal]
                if reference_node is not None:
                    best_idx = min(
                        range(len(remaining_nodes)),
                        key=lambda idx: abs(remaining_nodes[idx].last_contact - reference_node.last_contact),
                    )
                else:
                    best_idx = 0
                stage_assignments[stage_index][ordinal] = remaining_nodes.pop(best_idx)

        pipeline_infos: List[PipelineInfo] = []
        for ordinal, pipeline_index in enumerate(pipeline_indices):
            stage_metadata = pipeline_stage_metadata.get(pipeline_index, {})
            missing_stages = pipeline_missing_stage_map.get(pipeline_index, set())
            stages: List[StageInfo] = []
            for stage_index in range(n_stages):
                stage_name = stage_name_map[stage_index]
                assigned_node = stage_assignments[stage_index][ordinal]
                stage_nodes = [assigned_node] if assigned_node is not None else []
                stage_metadata_entry = stage_metadata.get(stage_name, {})
                stage_endpoint = stage_metadata_entry.get("endpoint")
                resolved_endpoint: Optional[str] = None
                if assigned_node is not None:
                    if stage_endpoint:
                        endpoint_ip_raw = self._extract_ip_from_endpoint(stage_endpoint)
                        endpoint_port = get_port(stage_endpoint)
                        endpoint_ip: Optional[str] = None
                        if endpoint_ip_raw:
                            try:
                                endpoint_ip_obj = ipaddress.ip_address(endpoint_ip_raw)
                                if not endpoint_ip_obj.is_unspecified:
                                    endpoint_ip = str(endpoint_ip_obj)
                            except ValueError:
                                endpoint_ip = None
                        if endpoint_ip and not assigned_node.address:
                            assigned_node.address = endpoint_ip
                        host = assigned_node.address or endpoint_ip
                        if host and endpoint_port is not None:
                            resolved_endpoint = f"{host}:{endpoint_port}"
                        elif host:
                            resolved_endpoint = host
                        elif endpoint_ip and endpoint_port is not None:
                            resolved_endpoint = f"{endpoint_ip}:{endpoint_port}"
                        elif endpoint_ip:
                            resolved_endpoint = endpoint_ip
                    else:
                        resolved_endpoint = assigned_node.address

                    if resolved_endpoint:
                        if resolved_endpoint not in assigned_node.endpoints:
                            assigned_node.endpoints.append(resolved_endpoint)
                stages.append(
                    StageInfo(
                        stage_index=stage_index,
                        stage_name=stage_name,
                        num_peers=len(stage_nodes),
                        nodes=stage_nodes,
                        missing=stage_name in missing_stages,
                        endpoint=resolved_endpoint,
                    )
                )
            pipeline_infos.append(
                PipelineInfo(
                    pipeline_index=pipeline_index,
                    stages=stages,
                )
            )

        return pipeline_infos

    def emit_log_entry(self, entry: LogEntry):
        logger.info(entry.as_console_message())
        if self.wandb_enabled:
            try:
                wandb.log(entry.to_wandb_payload())
            except Exception as e:
                logger.warning(f"Failed to log log entry to wandb: {e}")


    def run(self):
        dht = self.dht
        config = self.config
        experiment_prefix = config.experiment_prefix

        logger.info(f"MONITOR: Visible multiaddresses: {dht.get_visible_maddrs(latest=True)}")

        waiting_message = "MONITOR: Waiting for servers to join the training..."
        last_logged_step = -1
        waiting_for_metrics_logged = False
        while True:
            metrics_response = None
            try:
                if self.store_ip_addresses_path is not None:
                    with open(self.store_ip_addresses_path, "w") as f:
                        f.write(",".join(str(a) for a in self.dht.get_visible_maddrs(latest=True)))
                metrics_response = dht.get(experiment_prefix + "_metrics", latest=True)
                if metrics_response is None:
                    if not waiting_for_metrics_logged:
                        logger.info(waiting_message)
                        waiting_for_metrics_logged = True
                else:
                    metrics_dict = metrics_response.value
                    metrics = [
                        LocalMetrics.model_validate(entry.value)
                        for entry in metrics_dict.values()
                        if entry.value is not None
                    ]
                    if metrics:
                        waiting_for_metrics_logged = False
                        latest_step = max(item.step for item in metrics)
                        if latest_step != last_logged_step:
                            distributed_losses = [item.loss for item in metrics if item.trainer_id != -1]
                            baseline_losses = [item.loss for item in metrics if item.trainer_id == -1]

                            distributed_loss = (
                                sum(distributed_losses) / len(distributed_losses) if distributed_losses else None
                            )
                            baseline_loss = (
                                sum(baseline_losses) / len(baseline_losses) if baseline_losses else None
                            )

                            try:
                                pipeline_infos = self._collect_progress_info()
                            except Exception as e:
                                logger.warning(f"Error retrieving or processing progress: {e}")
                                import traceback
                                traceback.print_exc()
                                pipeline_infos = []

                            log_entry = LogEntry(
                                step=latest_step,
                                timestamp=time.time(),
                                distributed_loss=distributed_loss,
                                baseline_loss=baseline_loss,
                                pipelines=pipeline_infos,
                            )
                            self.emit_log_entry(log_entry)
                            last_logged_step = latest_step
                    else:
                        if not waiting_for_metrics_logged:
                            logger.info(waiting_message)
                            waiting_for_metrics_logged = True
                        logger.debug("MONITOR: Retrieved metrics response without valid entries.")
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                logger.warning(f"Error retrieving or processing metrics: {e}")
                import traceback
                traceback.print_exc()

            time.sleep(self.refresh_period)
    
    def shutdown(self):
        self.dht.shutdown()

def run_monitor(cfg: Config, refresh_period: int, store_ip_addresses_path: Optional[str]):
    monitor = Monitor(
        cfg,
        refresh_period=refresh_period,
        store_ip_addresses_path=store_ip_addresses_path,
    )
    
    logger.info(f"Experiment prefix: {cfg.experiment_prefix}")
    logger.info(f"Wandb project: {cfg.wandb_project}")
    logger.info(f"Refresh period: {refresh_period} seconds")
    logger.info(f"Store IP addresses path: {store_ip_addresses_path}")
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        logger.info("Monitor interrupted, shutting down...")
    finally:
        if monitor.wandb_enabled:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb: {e}")
        monitor.shutdown()


if __name__ == "__main__":
    parse_args_with_extra_kwargs = click.option("--refresh-period", type=int, default=5)(parse_args)
    parse_args_with_extra_kwargs = click.option("--store-ip-addresses-path", type=str, default=None)(parse_args_with_extra_kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*is used more than once. Remove its duplicate as parameters should be unique.*")
        cfg, extra_kwargs = parse_args_with_extra_kwargs(standalone_mode=False)
        run_monitor(
            cfg,
            **extra_kwargs
        )
