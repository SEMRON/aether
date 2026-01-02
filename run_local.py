import asyncio
import subprocess
import os
import tempfile
import yaml
import shutil
from pathlib import Path
from typing import Optional
import time
import click
from deepmerge import always_merger
from pydanclick import from_pydantic
from pydantic_yaml import parse_yaml_file_as

from hivemind.utils.logging import get_logger

from orchestrator import BaseOrchestrator
from run_script_utils import (
    run_monitor_proc,
    run_client_proc,
    wait_for_initial_peers,
    create_initial_peers_file,
    clear_data_server_log,
    run_baseline_model_trainer_proc,
    run_server_proc,
    wait_for_data_server_ready,
    get_public_ip,
    ensure_no_leftover_distqat_processes,
    is_wandb_logged_in,
)
from distqat.config import Config

if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ and os.environ["HF_HUB_ENABLE_HF_TRANSFER"] != "0":
    raise RuntimeError("HF_HUB_ENABLE_HF_TRANSFER is set to 1, please set it to 0 in your environment variables")




logger = get_logger(__name__)

DISABLE_QUANT = True

def _clear_dir_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except FileNotFoundError:
            continue


def _seed_checkpoint_dir(*, checkpoint_dir: Path, initial_checkpoint_path: Path) -> None:
    """
    Seed cfg.checkpoint_dir from a "golden" checkpoint so each run starts from the same weights,
    even if previous runs overwrote checkpoint_last.pt.

    Supported inputs:
    - File: copied to <checkpoint_dir>/baseline/checkpoint_last.pt
    - Directory:
      - If it looks like a full checkpoint root (contains baseline/ and/or expert subdirs with checkpoint_last.pt),
        copy the entire directory contents into <checkpoint_dir>
      - Otherwise, if it looks like a baseline checkpoint dir (contains checkpoint_last.pt), copy into
        <checkpoint_dir>/baseline
    """
    checkpoint_dir = Path(checkpoint_dir)
    initial_checkpoint_path = Path(initial_checkpoint_path)

    if initial_checkpoint_path.is_file():
        dst_dir = checkpoint_dir / "baseline"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / "checkpoint_last.pt"
        # Remove any existing file/symlink first (copy2 doesn't replace symlinks atomically).
        try:
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
        except FileNotFoundError:
            pass
        shutil.copy2(initial_checkpoint_path, dst_file, follow_symlinks=True)
        print(f"ORCHESTRATOR: Seeded baseline checkpoint from file: {initial_checkpoint_path} -> {dst_file}")
        return

    if not initial_checkpoint_path.is_dir():
        raise ValueError(f"--initial-checkpoint-path must be a file or directory, got: {initial_checkpoint_path}")

    candidate_full_root_baseline = initial_checkpoint_path / "baseline" / "checkpoint_last.pt"
    candidate_baseline_dir = initial_checkpoint_path / "checkpoint_last.pt"

    looks_like_full_root = False
    if candidate_full_root_baseline.exists():
        looks_like_full_root = True
    else:
        try:
            # Expert checkpoints are stored as <root>/<expert_uid>/checkpoint_last.pt
            for child in initial_checkpoint_path.iterdir():
                if child.is_dir() and (child / "checkpoint_last.pt").exists():
                    looks_like_full_root = True
                    break
        except FileNotFoundError:
            looks_like_full_root = False

    if looks_like_full_root:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _clear_dir_contents(checkpoint_dir)
        for item in initial_checkpoint_path.iterdir():
            dst = checkpoint_dir / item.name
            if item.is_dir() and not item.is_symlink():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst, follow_symlinks=True)
        print(f"ORCHESTRATOR: Seeded full checkpoint dir: {initial_checkpoint_path} -> {checkpoint_dir}")
        return

    if candidate_baseline_dir.exists():
        baseline_dst = checkpoint_dir / "baseline"
        _clear_dir_contents(baseline_dst)
        for item in initial_checkpoint_path.iterdir():
            dst = baseline_dst / item.name
            if item.is_dir() and not item.is_symlink():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst, follow_symlinks=True)
        print(f"ORCHESTRATOR: Seeded baseline checkpoint dir: {initial_checkpoint_path} -> {baseline_dst}")
        return

    raise FileNotFoundError(
        f"Could not find a checkpoint in {initial_checkpoint_path}. Expected one of: "
        f"{candidate_full_root_baseline} (full root) or {candidate_baseline_dir} (baseline dir)"
    )


class Orchestrator(BaseOrchestrator):
    def __init__(
        self,
        config_path: str,
        public_ip=None,
        num_servers=1,
        delete_checkpoints=False,
        no_baseline_trainer=False,
        initial_checkpoint_path: Optional[str] = None,
    ):
        super().__init__(
            config_path=config_path,
            public_ip=public_ip,
            disable_quant=DISABLE_QUANT,
        )
        self.num_servers = num_servers
        self.delete_checkpoints = delete_checkpoints
        self.no_baseline_trainer = no_baseline_trainer
        self.initial_checkpoint_path = initial_checkpoint_path
    async def start(self):
        # ensure_no_leftover_distqat_processes()

        if self.delete_checkpoints and self.initial_checkpoint_path is not None and self.config.checkpoint_dir is not None:
            seed_src_path = Path(self.initial_checkpoint_path)
            checkpoint_dir = Path(self.config.checkpoint_dir)
            try:
                seed_resolved = seed_src_path.resolve()
                ckpt_resolved = checkpoint_dir.resolve()
                seed_is_inside_ckpt_dir = (seed_resolved == ckpt_resolved) or (ckpt_resolved in seed_resolved.parents)
            except FileNotFoundError:
                seed_is_inside_ckpt_dir = False

            if seed_is_inside_ckpt_dir:
                raise RuntimeError(
                    f"--delete-checkpoints would delete the directory containing --initial-checkpoint-path.\n"
                    f"checkpoint_dir={checkpoint_dir}\n"
                    f"initial_checkpoint_path={seed_src_path}\n"
                    f"Move the seed checkpoint outside checkpoint_dir, or run without --delete-checkpoints."
                )

        # Ensure checkpoint_dir exists early (servers assert it is a directory when checkpointing is enabled)
        if self.config.checkpoint_dir is not None:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Delete checkpoints if requested
        if self.delete_checkpoints and self.config.checkpoint_dir is not None:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            if checkpoint_dir.exists():
                print(f"ORCHESTRATOR: Deleting checkpoint directory: {checkpoint_dir}")
                shutil.rmtree(checkpoint_dir)
                print(f"ORCHESTRATOR: Checkpoint directory deleted")
            else:
                print(f"ORCHESTRATOR: Checkpoint directory does not exist: {checkpoint_dir}")

        # Seed checkpoints from an initial checkpoint so each run starts from the same baseline.
        if self.initial_checkpoint_path is not None:
            if self.config.checkpoint_dir is None:
                raise RuntimeError("--initial-checkpoint-path requires Config.checkpoint_dir to be set")
            _seed_checkpoint_dir(
                checkpoint_dir=Path(self.config.checkpoint_dir),
                initial_checkpoint_path=Path(self.initial_checkpoint_path),
            )

        if not is_wandb_logged_in() and self.config.wandb_project:
            raise RuntimeError("Wandb is not logged in, please login to wandb using the wandb login command or set the wandb_project to None through the config file or command line argument")
        print("ORCHESTRATOR: Starting")

        initial_peers_path = create_initial_peers_file(log_dir=self.config.log_dir)

        # Start monitor
        self.monitor_proc = run_monitor_proc(config_path=self.config_path, refresh_period=2, store_ip_addresses_path=str(initial_peers_path), public_ip=self.public_ip, wandb_run_id=self.wandb_run_id)


        initial_peers_json = wait_for_initial_peers(initial_peers_path)


        # Ensure stale data_server.log does not cause a false-positive readiness
        ds_log_path = clear_data_server_log(log_dir=self.config.log_dir)
        
        # Spawn client, which in turn spawns trainers dynamically based on complete pipelines
        self.client_proc = run_client_proc(config_path=self.config_path, refresh_period=5, network_initial_peers=initial_peers_json, public_ip=self.public_ip, disable_quant=self.disable_quant)

        # Wait for data server to be ready (the client starts it)
        # We consider it ready when the log contains the startup line from start_manager(...)
        # wait_for_data_server_ready(client_proc=self.client_proc, ds_log_path=ds_log_path, deadline=20)


        # Spawn baseline model trainer
        if not self.no_baseline_trainer:
            print(f"ORCHESTRATOR: Spawning baseline model trainer")
            self.baseline_model_trainer_proc = run_baseline_model_trainer_proc(config_path=self.config_path, network_initial_peers=initial_peers_json, public_ip=self.public_ip, disable_quant=self.disable_quant, log_dir=self.config.log_dir)

        print(f"ORCHESTRATOR: Starting to spawn servers")

        # Spawn servers
        # for expert_idx in range(self.num_servers):
        #     for stage_idx,  pipeline_step_cfg in enumerate(self.config.model_pipeline.pipeline):
        #         _, stage = pipeline_step_cfg.model_name.split(".")
        #         proc = run_server_proc(config_path=self.config_path, network_initial_peers=initial_peers_json, public_ip=self.public_ip, idx=expert_idx, stage_index=stage_idx, disable_quant=self.disable_quant)
        #         self.server_procs[f"server_{stage}_{expert_idx}"] = proc
        #         time.sleep(3)
        # Spawn servers
        for idx in range(self.num_servers):
            # Pass only what's necessary, config file handles the rest
            self.server_procs[f"server_{idx}"] = run_server_proc(
                config_path=self.config_path, 
                network_initial_peers=initial_peers_json, 
                public_ip=self.public_ip, 
                idx=idx, 
                stage_index=0 if len(self.config.model_pipeline.pipeline) == 1 else None,
                disable_quant=self.disable_quant,
                wandb_run_id=self.wandb_run_id
            )
            # time.sleep(3)

        print(f"ORCHESTRATOR: Servers spawned")

    async def wait(self):
        print("ORCHESTRATOR: Waiting for children to finish")

        # Wait for the client process, which manages the distributed trainers
        # Finished training when all trainers are completed
        while True:
            if self.client_proc.poll() is not None:
                print(f"ORCHESTRATOR: Client exited with code {self.client_proc.returncode}")
                break
            
            done = {n: p for n, p in self.trainer_procs.items() if p.poll() is not None}
            for label, p in done.items():
                print(f"ORCHESTRATOR: Trainer {label} exited with code {p.returncode}")
            
            try:
                self.client_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                pass

@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--public-ip", type=str, default=None)
@click.option("--config-path", type=str, default="configs/resnet18.yaml")
@click.option("--num-servers", type=int, default=1)
@click.option("--delete-checkpoints", is_flag=True, default=False, help="Delete checkpoint directory before starting the run")
@click.option("--no-baseline-trainer", is_flag=True, default=False, help="Disable baseline trainer")
@click.option(
    "--initial-checkpoint-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=str),
    default=None,
    help=(
        "Seed <checkpoint_dir> from this checkpoint before starting, so each run starts from the same pretrained "
        "weights even if previous runs overwrote checkpoint_last.pt. "
        "Pass a checkpoint file to seed <checkpoint_dir>/baseline/checkpoint_last.pt, or a directory to seed either "
        "the full checkpoint tree or the baseline directory."
    ),
)
@from_pydantic(Config)
def main(
    public_ip: Optional[str],
    config_path: str,
    num_servers: int,
    delete_checkpoints: bool,
    no_baseline_trainer: bool,
    initial_checkpoint_path: Optional[str],
    config: Config,
    **_kwargs,
):
    """Run the same local orchestrator flow as the old argparse entrypoint."""
    cfg = parse_yaml_file_as(Config, config_path)
    base_dict = cfg.model_dump(exclude_unset=True)
    override_dict = config.model_dump(exclude_unset=True)
    merged_dict = always_merger.merge(base_dict, override_dict)
    merged_cfg = cfg.model_validate(merged_dict)

    merged_cfg.world_size = max(1, num_servers)

    resolved_public_ip = public_ip or get_public_ip()
    print("Public IP:", resolved_public_ip)

    fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="distqat_config_")
    os.close(fd)

    try:
        with open(temp_config_path, "w") as temp_config_file:
            yaml.dump(merged_cfg.model_dump(mode="json"), temp_config_file)

        async def run():
            orch = Orchestrator(
                config_path=temp_config_path,
                public_ip=resolved_public_ip,
                num_servers=num_servers,
                delete_checkpoints=delete_checkpoints,
                no_baseline_trainer=no_baseline_trainer,
                initial_checkpoint_path=initial_checkpoint_path,
            )
            try:
                await orch.start()
                await orch.wait()
                print("Training finished. Shutting down...")
            except KeyboardInterrupt:
                print("Keyboard interrupt. Shutting down...")
            finally:
                await orch.shutdown()

        asyncio.run(run())
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass