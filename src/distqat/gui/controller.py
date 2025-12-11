import asyncio
import re
import logging
from typing import Dict, Optional, Callable, List
from pathlib import Path
from collections import deque
from datetime import datetime
import wandb

from .config import GlobalState, Server, JobConfig
from .ssh_runner import SSHSession
import json

logger = logging.getLogger(__name__)

class Controller:
    def __init__(self, no_wandb: bool = False):
        self.no_wandb = no_wandb
        self.state = GlobalState.load_from_file()
        self.sessions: Dict[str, SSHSession] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        
        # Runtime state
        self.initial_peers: Optional[str] = None
        self.head_server: Optional[Server] = None
        self.active_workers: List[Server] = []
        self.wandb_run_url: Optional[str] = None
        self.wandb_run_path: Optional[str] = None
        if not self.no_wandb:
            self.wandb_api = wandb.Api()
        
        # Callbacks for UI updates
        self.log_callback: Optional[Callable[[str, str], None]] = None # (server_name, message) -> None
        self.status_callback: Optional[Callable[[str], None]] = None
        self.server_status_callback: Optional[Callable[[str, bool], None]] = None # (server_name, is_running) -> None
        self.error_callback: Optional[Callable[[dict], None]] = None # (error_event) -> None
        self.error_reset_callback: Optional[Callable[[str], None]] = None # (source) -> None
        
        # Error monitoring
        self.log_buffers: Dict[str, deque] = {} # source -> deque(maxlen=10)
        self.error_patterns = [
            r"error", r"exception", r"traceback", r"failed", r"fatal"
        ]
        self.compiled_error_patterns = [re.compile(p, re.IGNORECASE) for p in self.error_patterns]
        
        self.file_last_scanned: Dict[str, tuple] = {} # filename -> (updated_at_timestamp, size_str)

    def _update_log_buffer(self, source: str, line: str):
        if source not in self.log_buffers:
            self.log_buffers[source] = deque(maxlen=20) # Keep last 20 lines for context
        self.log_buffers[source].append(line)

    def _check_line_for_error(self, source: str, line: str):
        # Update buffer first so the current line is included in context
        self._update_log_buffer(source, line)
        
        # Check against patterns
        if any(p.search(line) for p in self.compiled_error_patterns):
            # Capture context
            context = list(self.log_buffers[source])
            
            error_event = {
                "source": source,
                "message": line.strip(),
                "context": "\n".join(context),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
            if self.error_callback:
                self.error_callback(error_event)

    def scan_text_for_errors(self, text: str, source: str):
        """Scan a block of text for errors."""
        # Reset errors for this source to avoid duplicates on rescan
        if self.error_reset_callback:
            self.error_reset_callback(source)

        if source in self.log_buffers:
            self.log_buffers[source].clear()
            
        for line in text.splitlines():
            self._check_line_for_error(source, line)

    async def check_wandb_files_for_errors(self, files: List[Dict]):
        """Passively check provided list of wandb files for updates and scan them."""
        for f in files:
            name = f['name']
            updated_at = f['updated_at']
            size = f['size']
            
            # If new or updated (check both timestamp and size)
            last_scan = self.file_last_scanned.get(name)
            current_sig = (updated_at, size)
            
            if last_scan != current_sig:
                try:
                    # Download content
                    # Note: downloading large files frequently might be expensive
                    print(f"Scanning updated file: {name} (Size: {size})")
                    content = await self.get_file_content(f['url'])
                    
                    if content.startswith("Error"):
                        logger.error(f"Failed to download {name} for scanning: {content}")
                        continue
                        
                    self.scan_text_for_errors(content, f"File: {name}")
                    self.file_last_scanned[name] = current_sig
                except Exception as e:
                    logger.error(f"Error scanning file {name}: {e}")

    def save_state(self):
        self.state.save_to_file()

    def add_server(self, server: Server):
        self.state.servers.append(server)
        self.save_state()

    def remove_server(self, server_name: str):
        self.state.servers = [s for s in self.state.servers if s.name != server_name and s.hostname != server_name]
        self.save_state()

    def get_server(self, name: str) -> Optional[Server]:
        for s in self.state.servers:
            if s.display_name == name:
                return s
        return None

    async def stop_task(self, task_id: str):
        """Stop a specific task/session."""
        if task_id in self.tasks:
            self.tasks[task_id].cancel()
            del self.tasks[task_id]
        
        if task_id in self.sessions:
            await self.sessions[task_id].close()
            del self.sessions[task_id]
            
        if self.status_callback:
            self.status_callback(f"Stopped task: {task_id}")

    async def stop_all(self):
        """Stop all running sessions."""
        if self.status_callback:
            self.status_callback("Stopping all processes...")

        for name, task in self.tasks.items():
            print(f"Cancelling task {name}")
            task.cancel()

        pkill_tasks = []
        
        active_servers = {}
        for sess in self.sessions.values():
            if sess.server.hostname not in active_servers:
                active_servers[sess.server.hostname] = sess.server
                
        for server in active_servers.values():
            async def run_pkill(server=server):
                try:
                    sess = SSHSession(server)
                    await sess.connect()
                    await sess.run_command("pkill -f distqat", lambda msg: None)
                    await sess.close()
                except Exception as e:
                    logger.error(f"Failed to pkill on {server.display_name}: {e}")
            pkill_tasks.append(run_pkill())

        if pkill_tasks:
            await asyncio.gather(*pkill_tasks, return_exceptions=True)
        

        stop_coros = []
        for sess in self.sessions.values():
            stop_coros.append(sess.close())

        if stop_coros:
            await asyncio.gather(*stop_coros, return_exceptions=True)

        self.sessions.clear()
        self.tasks.clear()
        self.initial_peers = None
        self.head_server = None
        self.active_workers = []
        self.wandb_run_url = None
        self.wandb_run_path = None

        if self.status_callback:
            self.status_callback("All processes stopped.")

    async def reset_all_processes(self):
        """Force kill distqat processes on all configured servers."""
        if self.status_callback:
            self.status_callback("Resetting all processes on all servers...")

        reset_tasks = []
        for server in self.state.servers:
            async def run_reset(s=server):
                try:
                    sess = SSHSession(s)
                    await sess.connect()
                    # We use a lambda for log callback that does nothing
                    await sess.run_command("pkill -f distqat", lambda msg: None)
                    await sess.close()
                    logger.info(f"Reset processes on {s.display_name}")
                except Exception as e:
                    logger.error(f"Failed to reset processes on {s.display_name}: {e}")
            
            reset_tasks.append(run_reset())

        if reset_tasks:
            await asyncio.gather(*reset_tasks, return_exceptions=True)

        # Clear local state as well since we killed everything
        await self.stop_all()

        if self.status_callback:
            self.status_callback("Reset command sent to all servers.")

    async def get_wandb_metrics(self, keys: Optional[List[str]] = None) -> Dict[str, List]:
        """Fetch metrics history from WandB."""
        if self.no_wandb or not self.wandb_run_path or not self.wandb_api:
            return {}
            
        try:
            # Run in thread to avoid blocking asyncio loop
            def _fetch():
                try:
                    run = self.wandb_api.run(self.wandb_run_path)
                    # Fetch history, limiting to last 500 points for responsiveness
                    # samples=500 is a good default if supported, or we can slice
                    # run.history returns a list of dicts
                    history = run.history(keys=keys, pandas=False, samples=1000) 
                    return history
                except Exception as e:
                    logger.error(f"WandB API error: {e}")
                    return []
            
            history = await asyncio.to_thread(_fetch)
            
            if not history:
                return {}

            # Process into dict of lists
            # Collect all keys found in history
            all_keys = set()
            for row in history:
                all_keys.update(row.keys())
            
            result = {k: [] for k in all_keys}
            
            for row in history:
                for k in all_keys:
                    result[k].append(row.get(k))
                    
            return result
        except Exception as e:
            logger.error(f"Failed to fetch WandB metrics: {e}")
            return {}

    async def get_wandb_files(self) -> List[Dict[str, str]]:
        """Fetch list of files uploaded to WandB run."""
        if self.no_wandb or not self.wandb_run_path or not self.wandb_api:
            return []
            
        try:
            def _fetch():
                try:
                    run = self.wandb_api.run(self.wandb_run_path)
                    return [{"name": f.name, "url": f.url, "size": str(f.size), "updated_at": f.updated_at} for f in run.files() if f.name.startswith("logs/")]
                except Exception as e:
                     logger.error(f"WandB API error: {e}")
                     return []
            
            files = await asyncio.to_thread(_fetch)
            return files
        except Exception as e:
            logger.error(f"Failed to fetch WandB files: {e}")
            return []

    async def get_file_content(self, url: str) -> str:
        """Download and return content of a file from URL."""
        if self.no_wandb or not self.wandb_api.api_key:
             return "Error: No WandB API key found."

        import aiohttp
        try:
            auth = None
            if "api.wandb.ai" in url:
                 auth = aiohttp.BasicAuth('api', self.wandb_api.api_key)

            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        return text
                    else:
                        return f"Error: HTTP {response.status}"
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return f"Error downloading file: {e}"

    async def start_head_node(self, server_name: str, config: JobConfig):
        """Start the head node (client/monitor)."""
        server = self.get_server(server_name)
        if not server:
            raise ValueError(f"Server {server_name} not found")
            
        self.head_server = server
        self.state.last_job_config = config
        self.save_state()
        
        # Prepare command
        # Assuming the repo is cloned at remote_root_dir
        env_parts = [f"cd {server.remote_root_dir}", "source .venv/bin/activate"]

        if getattr(config, "hf_token", None):
            env_parts.append(f"export HF_TOKEN={config.hf_token}")

        cmd = " && ".join(env_parts) + f" && python start_trainer_client.py --config-path {config.config_path} --public-ip {server.hostname}"
        if config.wandb_api_key and not self.no_wandb:
            self.wandb_api = wandb.Api(api_key=config.wandb_api_key)
            cmd = f"export WANDB_API_KEY={config.wandb_api_key} && " + cmd
        
        session = SSHSession(server)
        self.sessions[server.display_name] = session
        print(f"Starting head node on {server.display_name} with command: {cmd}")
        
        # Create a wrapper callback to parse for initial peers
        def _head_log_handler(line: str):
            line = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line)
            if line.strip() == 'm':
                return

            if self.log_callback:
                self.log_callback(server.display_name, line)
            
            # Check for errors
            self._check_line_for_error(server.display_name, line)
            
            # Check for peers
            # Expected format: "Initial peers JSON: [\"/ip4/...\"]"
            if "Initial peers JSON:" in line:
                try:
                    json_part = line.split("Initial peers JSON: ", 1)[1].strip()
                    peers_list = json.loads(json_part)
                    
                    if peers_list:
                        peer = peers_list[0]
                        if server.mapped_monitor_port:
                            # Swap port
                            peer = re.sub(r'/tcp/\d+/', f'/tcp/{server.mapped_monitor_port}/', peer)
                            print(f"Mapped initial peer to port {server.mapped_monitor_port}: {peer}")
                        
                        self.initial_peers = peer

                    if self.status_callback:
                        self.status_callback(f"Received initial peers from {server.display_name}")
                    print(f"Initial peers: {self.initial_peers}")
                except Exception as e:
                    logger.error(f"Failed to parse initial peers: {e}")
            
            # Check for WandB URL
            # Expected format might vary, but usually: "wandb: View run at https://wandb.ai/..."
            if "View run at https://wandb.ai/" in line:
                try:
                     url = line.split("View run at ", 1)[1].strip()
                     # Remove potential trailing chars or ANSI codes if any (basic cleanup)
                     url = url.split()[0] 
                     self.wandb_run_url = url
                     
                     # Extract run path from URL: https://wandb.ai/entity/project/runs/run_id
                     # path = entity/project/run_id
                     try:
                         parts = url.split("wandb.ai/")[1].split("/runs/")
                         entity_project = parts[0]
                         run_id = parts[1]
                         self.wandb_run_path = f"{entity_project}/{run_id}"
                         print(f"WandB Run Path: {self.wandb_run_path}")
                     except Exception:
                         logger.warning(f"Could not extract run path from {url}")

                     print(f"WandB Run URL: {self.wandb_run_url}")
                except Exception as e:
                    logger.error(f"Failed to parse WandB URL: {e}")

        def _head_status_handler(is_running: bool, s_name=server_name):
            pass # Currently we don't update head status in UI, but we could

        self.tasks[server.display_name] = asyncio.create_task(
            session.run_command(cmd, _head_log_handler, _head_status_handler)
        )
        
        if self.status_callback:
            self.status_callback(f"Head node started on {server.display_name}")

    async def start_worker_nodes(self, server_names: List[str], num_servers_per_node: int = 1, device: str = "cpu", batch_size: int = 32, inner_steps: int = 500, grpc_announce_port: Optional[int] = None, host_port: Optional[int] = None):
        """Start worker nodes (servers)."""
        if not self.initial_peers:
            raise RuntimeError("Initial peers not yet received. Start head node first.")
            
        print(f"Starting worker nodes: {server_names} with {device} batch_size={batch_size} inner_steps={inner_steps} grpc_announce_port={grpc_announce_port} host_port={host_port}")
        for name in server_names:
            print(f"Starting worker node: {name}")
            server = self.get_server(name)
            if not server:
                print(f"Server not found: {name}")
                continue
                
            # Generate unique session ID for multiple processes on same server
            import uuid
            session_id = f"{name}_{str(uuid.uuid4())[:8]}"
            
            # Prepare command
            # We need to escape quotes in the JSON for the shell
            # The JSON string from stdout often comes double-quoted or with outer brackets that are part of the string representation.
            
            clean_peers = self.initial_peers.strip()
            
            # Handle potential double encoding from previous steps
            # e.g. "[\"/ip4/...\"]" -> ["/ip4/..."]
            try:
                if clean_peers.startswith('"') and clean_peers.endswith('"'):
                    clean_peers = json.loads(clean_peers)
            except Exception:
                pass

            # Now try to parse as list to confirm valid JSON
            try:
                peers_list = json.loads(clean_peers)
                if isinstance(peers_list, str): # Handle case where it was double-encoded JSON string
                     peers_list = json.loads(peers_list)
                
                # Re-dump to ensure minimal, valid JSON string
                final_json = json.dumps(peers_list)
                peers_arg = f"'{final_json}'"
            except Exception as e:
                logger.warning(f"Failed to parse peers JSON cleanly: {e}. Using raw string.")
                # Fallback: try to strip outer quotes if it looks like a stringified list
                if clean_peers.startswith('"') and clean_peers.endswith('"'):
                     clean_peers = clean_peers[1:-1].replace('\\"', '"')
                peers_arg = f"'{clean_peers}'"

            print(f"Peers arg: {peers_arg}")

            env_parts = [f"cd {server.remote_root_dir}", "source .venv/bin/activate"]
            # Optional HuggingFace token (comes from last_job_config)
            if getattr(self.state.last_job_config, "hf_token", None):
                env_parts.append(f"export HF_TOKEN={self.state.last_job_config.hf_token}")
            env_parts.append("export HF_HUB_ENABLE_HF_TRANSFER=0")

            env_prefix = " && ".join(env_parts)
            cmd = f"{env_prefix} && python start_servers.py --config-path {self.state.last_job_config.config_path} --network-initial-peers {peers_arg} --num-servers {num_servers_per_node} --public-ip {server.hostname} --device {device} --diloco-batch-size-per-step {batch_size} --diloco-inner-steps {inner_steps}"
            
            if host_port:
                cmd += f" --network-server-base-hostport-announce {host_port}"

            if grpc_announce_port:
                cmd += f" --network-server-base-grpc-announceport {grpc_announce_port}"

            if self.state.last_job_config.wandb_api_key and not self.no_wandb:
                 cmd = f"export WANDB_API_KEY={self.state.last_job_config.wandb_api_key} && " + cmd
            
            session = SSHSession(server)
            self.sessions[session_id] = session
            
            def _worker_log_handler(line: str, s_name=session_id):
                print(f"Worker log ({s_name}): {line}")
                line = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line)
                if line.strip() == 'm':
                    return
                if self.log_callback:
                    self.log_callback(s_name, line)
                
                # Check for errors
                self._check_line_for_error(s_name, line)

            def _worker_status_handler(is_running: bool, s_name=name):
                if self.server_status_callback:
                    self.server_status_callback(s_name, is_running)

            print(f"Starting worker node on {name} (Session {session_id}) with command: {cmd}")
            self.tasks[session_id] = asyncio.create_task(
                session.run_command(cmd, _worker_log_handler, _worker_status_handler)
            )
            
        if self.status_callback:
             self.status_callback(f"Started {len(server_names)} worker nodes.")

