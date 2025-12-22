import sys, subprocess
from typing import Optional
import json
from pathlib import Path        
import time
import urllib.request
import os
import shutil
import wandb
from distqat.config import Config, NetworkConfig
from pydantic_yaml import parse_yaml_file_as

ROOT_DIR = Path(__file__).parent

def get_public_ip():
    """
    Best-effort attempt to discover the machine's public IPv4 address.

    Returns:
        str: The detected public IP address.

    Raises:
        RuntimeError: If all detection methods fail.
    """
    services = [
        "https://checkip.amazonaws.com",
        "https://ipv4.icanhazip.com",
        "https://api.ipify.org",
    ]
    for service in services:
        try:
            with urllib.request.urlopen(service, timeout=3) as response:
                ip = response.read().decode().strip()
                if ip:
                    return ip
        except Exception:
            continue
    raise RuntimeError("Unable to determine public IP address from external services. Please set the public IP manually by using the --public-ip flag.")

def get_monitor_addresses(public_ip: str, network_config: NetworkConfig):
    port = network_config.monitor_port
    hostmaddr_str = f'["/ip4/0.0.0.0/tcp/{port}"]'
    announcemaddr_str = f'["/ip4/{public_ip}/tcp/{port}", "/ip4/127.0.0.1/tcp/{port}"]'
    return hostmaddr_str, announcemaddr_str


def get_client_addresses(public_ip: str, network_config: NetworkConfig):
    port = network_config.client_port
    hostmaddr_str = f'["/ip4/0.0.0.0/tcp/{port}"]'
    announcemaddr_str = f'["/ip4/{public_ip}/tcp/{port}", "/ip4/127.0.0.1/tcp/{port}"]'
    return hostmaddr_str, announcemaddr_str


def get_server_addresses(public_ip: str, network_config: NetworkConfig, idx: int = 0):
        hostport = network_config.server_base_hostport + idx
        hostport_announce = network_config.server_base_hostport_announce + idx
        grpcport = network_config.server_base_grpcport + idx
        grpc_announceport = network_config.server_base_grpc_announceport + idx

        listen_on = f"0.0.0.0:{grpcport}"
        announce_endpoint = f"{public_ip}:{grpc_announceport}"

        # host_maddrs uses the local binding port
        hostmaddr_str = f'["/ip4/0.0.0.0/tcp/{hostport}"]'
        # announce_maddrs uses the mapped/announced port for DHT peer discovery
        announcemaddr_str = f'["/ip4/{public_ip}/tcp/{hostport_announce}", "/ip4/127.0.0.1/tcp/{hostport}"]'
        
        return hostmaddr_str, announcemaddr_str, listen_on, announce_endpoint


def spawn_process(cmd, logfile):
    return subprocess.Popen(cmd, stdout=logfile, stderr=logfile, text=True)


def create_initial_peers_file(log_dir: Path):
    initial_peers_path = log_dir / "initial_peers.txt"
    initial_peers_path.parent.mkdir(parents=True, exist_ok=True)
    initial_peers_path.touch()
    return initial_peers_path


def wait_for_initial_peers(initial_peers_path: str):
    initial_peers = None
    initial_peers_update_time = initial_peers_path.stat().st_mtime
    while True:
        curr_time = initial_peers_path.stat().st_mtime 
        if curr_time > initial_peers_update_time:
            with open(initial_peers_path, "r") as f:
                initial_peers = [p.strip() for p in f.read().split(",") if p.strip()]
            break
        time.sleep(0.5)
    assert initial_peers is not None

    if len(initial_peers) == 0:
        error_message = "--------------------------------\nNo initial peers found, try running again\n--------------------------------"
        raise RuntimeError(error_message)

    initial_peers_json = json.dumps(initial_peers)
    print(f"\n--------------------------------\nInitial peers JSON: {initial_peers_json}\n--------------------------------\n")

    return initial_peers_json


def clear_data_server_log(log_dir: Path):
    ds_log_path = log_dir / "data_server.log"
    try:
        if ds_log_path.exists():
            ds_log_path.unlink()
    except Exception:
        pass

    return ds_log_path


def wait_for_data_server_ready(client_proc: subprocess.Popen, ds_log_path: str, deadline: int = 60):
    ready_line = "Data server manager started"
    ds_deadline = time.time() + deadline  # up to 60s for large models (e.g., GPT-Neo) to initialize
    print(f"\n--------------------------------\nWaiting for data server to be ready... \n--------------------------------")
    ready = False
    while time.time() < ds_deadline:
        try:
            if ds_log_path.exists():
                with open(ds_log_path, "r") as f:
                    contents = f.read()
                    if ready_line in contents:
                        ready = True
                        break
        except Exception as e:
            print(f"Error checking data server log: {e}")
            pass
        # If client died, stop waiting
        if client_proc.poll() is not None:
            print("Client died, stopping data server wait")
            raise RuntimeError("Client exited before data server manager became ready")
        time.sleep(0.5)
    if not ready:
        raise RuntimeError("Timed out waiting for data server manager to start, check the log file for more details in logs/<experiment_prefix>/data_server.log")
    
    print(f"\n--------------------------------\nData server should be ready now \n--------------------------------")


def ensure_no_leftover_distqat_processes():
    """Raise if distqat processes other than the current one are still running."""
    pgrep = shutil.which("pgrep")
    if not pgrep:
        return

    result = subprocess.run([pgrep, "-af", "distqat/"], capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        # Unable to determine; assume safe to continue.
        return

    leftover = []
    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        leftover.append(line[:100] + "..." if len(line) > 100 else line)

    if not leftover:
        return

    displayed = leftover[:5]
    message_lines = [
        "\n--------------------------------",
        "Detected leftover distqat processes from a previous run.",
        "Please kill them (for example `pkill -f distqat`, or Stop and Reset buttons in the GUI) before starting a new run.",
        "Leftover processes:",
        *displayed,
    ]
    if len(leftover) > len(displayed):
        message_lines.append(f"... plus {len(leftover) - len(displayed)} more")

    raise RuntimeError("\n".join(message_lines))


def run_monitor_proc(config_path: str, refresh_period: int, store_ip_addresses_path: str, public_ip: Optional[str] = None, wandb_run_id: Optional[str] = None):
    config = parse_yaml_file_as(Config, config_path)
    monitor_cmd = [
        sys.executable, ROOT_DIR / "src/distqat/distributed/monitor.py",
        "--config-path", config_path,
        "--refresh-period", str(refresh_period),
        "--store-ip-addresses-path", store_ip_addresses_path,
        "--wandb-run-id", wandb_run_id,
    ]
    if public_ip:
        hostmaddr, announcemaddr = get_monitor_addresses(public_ip=public_ip, network_config=config.network)
        monitor_cmd.extend(["--network-host-maddrs", hostmaddr])
        monitor_cmd.extend(["--network-announce-maddrs", announcemaddr])

    monitor_proc = spawn_process(monitor_cmd, logfile=None)
    return monitor_proc


def run_client_proc(config_path: str, refresh_period: int, network_initial_peers: str, public_ip: Optional[str] = None, disable_quant: bool = False, wandb_run_id: Optional[str] = None):
    config = parse_yaml_file_as(Config, config_path)
    client_cmd = [
        sys.executable, ROOT_DIR / "src/distqat/distributed/client.py",
        "--config-path", config_path,
        "--refresh-period", str(refresh_period),
        "--network-initial-peers", network_initial_peers,
    ]
    if public_ip:
        client_cmd.extend(["--public-ip", public_ip])
        hostmaddr, announcemaddr = get_client_addresses(public_ip=public_ip, network_config=config.network)
        client_cmd.extend(["--network-host-maddrs", hostmaddr])
        client_cmd.extend(["--network-announce-maddrs", announcemaddr])
    if disable_quant:
        client_cmd.append("--disable-quant")
    if wandb_run_id:
        client_cmd.extend(["--wandb-run-id", wandb_run_id])
    return spawn_process(client_cmd, logfile=None)


def run_server_proc(config_path: str, network_initial_peers: str, public_ip: Optional[str] = None, idx: int = 0, stage_index: Optional[int] = None, disable_quant: bool = False, device: Optional[str] = None, diloco_batch_size_per_step: Optional[int] = None, wandb_run_id: Optional[str] = None):
    config = parse_yaml_file_as(Config, config_path)
    server_cmd = [
        sys.executable, ROOT_DIR / "src/distqat/distributed/server.py",
        "--config-path", config_path,
        "--network-initial-peers", network_initial_peers,
    ]
    if public_ip:
        hostmaddr, announcemaddr, listen_on, announce_endpoint = get_server_addresses(public_ip=public_ip, network_config=config.network, idx=idx)
        server_cmd.extend(["--network-host-maddrs", hostmaddr])
        server_cmd.extend(["--network-announce-maddrs", announcemaddr])
        # Bind to all interfaces, but announce the public endpoint for cross-machine access
        server_cmd.extend(["--listen-on", listen_on])
        server_cmd.extend(["--announce-endpoint", announce_endpoint])
    if disable_quant:
        server_cmd.append("--disable-quant")
    if device:
        server_cmd.extend(["--device", device])
    if diloco_batch_size_per_step:
        server_cmd.extend(["--diloco-batch-size-per-step", str(diloco_batch_size_per_step)])
    if stage_index is not None:
        server_cmd.extend(["--expert-index", str(idx)])
        server_cmd.extend(["--stage-index", str(stage_index)])
    if wandb_run_id:
        server_cmd.extend(["--wandb-run-id", wandb_run_id])
    return spawn_process(server_cmd, logfile=None)


def run_baseline_model_trainer_proc(config_path: str, network_initial_peers: str, public_ip: Optional[str] = None, disable_quant: bool = False, log_dir: Path = None):
    baseline_cmd = [
        sys.executable, ROOT_DIR / "src/distqat/distributed/trainer.py",
        "--run-locally",
        "--trainer-id", "-1",
        "--config-path", config_path,
        "--network-initial-peers", network_initial_peers,
    ]
    if disable_quant:
        baseline_cmd.append("--disable-quant")
    log_file = open(log_dir / f"baseline_model_trainer.log", "w")
    return spawn_process(baseline_cmd, logfile=log_file)


def is_wandb_logged_in():
    try:
        wandb.login(key=None, relogin=False)
        return True
    except Exception:
        return False