import signal
import warnings
import threading
from typing import Optional
from pathlib import Path
import time
import random

import torch
# Avoid exhausting file descriptors under heavy tensor sharing (hivemind/torch mp reduction).
torch.multiprocessing.set_sharing_strategy("file_system")

from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.dht import DHT
from hivemind.utils import get_dht_time

from distqat.distributed.server.server import SwarmServer
from distqat.distributed.utils.auto_discovery import generate_expert_and_stage_idx
from distqat.config import Config, parse_args
from distqat.optimizers import get_diloco_optimizer_cls_kwargs
from distqat.distributed.utils.reassignment_monitor import ReassignmentMonitorThread
from distqat.utils import logging
from distqat.utils.compression import get_compression_kwargs

import subprocess
import sys
import json
import wandb

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

def start_trainer(cfg: Config, expert_index: int):
    logger.info(f"Starting trainer for expert {expert_index}")
    logger.info(f"Initial peers: {cfg.network.initial_peers}")
    initial_peers_json = json.dumps(cfg.network.initial_peers)
    cmd = [
        sys.executable, "src/distqat/distributed/trainer.py",
        "--trainer-id", str(expert_index),
        "--config-path", cfg.path,
        "--network-initial-peers", initial_peers_json,
    ]


    log_path = cfg.log_dir / f"trainer_{expert_index}.log"
    log_file = open(log_path, "w")
    try:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
        logger.info(f"Spawned trainer {expert_index} with PID {proc.pid}")
        logging.track_log_file(log_path)
        return proc
    except Exception as e:
        logger.error(f"Failed to spawn trainer {expert_index}: {e}")
        log_file.close()
        return None


def start_trainer_inprocess(*, cfg: Config, expert_index: int, dht: DHT, local_expert_backends):
    """
    Start the trainer in the same process (background thread) to allow bypassing gRPC for local experts.

    This is opt-in because it changes failure isolation vs spawning a subprocess.
    """
    from distqat.distributed.trainer import SwarmTrainer

    def _run():
        trainer = SwarmTrainer(
            trainer_id=expert_index,
            config=cfg,
            use_baseline_model=False,
            dht=dht,
            local_expert_backends=local_expert_backends,
        )
        try:
            trainer.train()
        finally:
            trainer.shutdown()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    logger.info(f"Started in-process trainer thread for expert {expert_index}")
    return t

def start_server(
    cfg: Config,
    stage_index: Optional[int] = None,
    expert_index: Optional[int] = None,
    listen_on: str = "0.0.0.0:*",
    announce_endpoint: Optional[str] = None,
    trainer_in_process: bool = False,
):
    # Create DHT for expert discovery
    dht = DHT(
        start=True,
        initial_peers=cfg.network.initial_peers,
        host_maddrs=cfg.network.host_maddrs,
        announce_maddrs=cfg.network.announce_maddrs,
    ) 
    
    
    if stage_index is None or expert_index is None:
        logger.info("Auto-discovering pipeline gaps...")
        expert_index, stage_index = generate_expert_and_stage_idx(dht, cfg)
    
    pipeline_length = len(cfg.model_pipeline.pipeline)
    if stage_index < 0 or stage_index >= pipeline_length:
        raise ValueError(f"Stage index {stage_index} is out of range, expected range is [0, {pipeline_length - 1}]")

    
    wandb_enabled = cfg.wandb_project is not None

    if wandb_enabled:
        # Prefer CLI-provided wandb_run_id, otherwise try to retrieve from DHT with retries
        if cfg.wandb_run_id:
            logger.info(f"Using CLI-provided wandb_run_id: {cfg.wandb_run_id}")
        else:
            dht_run_id = logging.get_wandb_run_id_with_retries(
                dht, cfg.experiment_prefix, max_retries=15, retry_delay=2.0
            )
            if dht_run_id:
                cfg.wandb_run_id = dht_run_id
                logger.info(f"Retrieved wandb_run_id from DHT: {dht_run_id}")
            else:
                logger.warning("wandb_run_id not found in DHT after retries and not provided via CLI. Wandb may create separate runs.")

        try:
            wandb_kwargs = dict(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=f"{cfg.experiment_prefix}",
            )
            if cfg.wandb_run_id:
                wandb_kwargs["id"] = cfg.wandb_run_id
                wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize wandb for server: {e}. Continuing without wandb logging.")
            wandb_enabled = False
    

    log_file = cfg.log_dir / f"server_{expert_index}_{stage_index}.log"

    logging.setup_file_logging(log_file, wandb_enabled=wandb_enabled)

    models, stages = [], []
    for pipeline_step_cfg in cfg.model_pipeline.pipeline:
        model, stage = pipeline_step_cfg.model_name.split(".")
        models.append(model)
        stages.append(stage)

    expert_cls = f"{models[stage_index]}.{stages[stage_index]}"
    expert_uid = f"{stages[stage_index]}.0.{expert_index}.0"

    run_id = f"{cfg.experiment_prefix}_{stage_index}"
    compression = get_compression_kwargs(cfg.network.hivemind_compression)

    optim_cls, optim_kwargs = get_diloco_optimizer_cls_kwargs(run_id, cfg.diloco, compression)
    
    visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
    logger.info(f"Running DHT node on {visible_maddrs_str}")
    logger.debug(f"Initial peers: {cfg.network.initial_peers}")
    logger.info(f"====> RUN_ID: {optim_kwargs['run_id']}")

    stage_cfg = cfg.model_pipeline.pipeline[stage_index]
    hidden_dim = stage_cfg.hid_dim

    autocast_dtype = None
    if cfg.data.precision in ("fp16-mixed", "bf16-mixed"):
        autocast_dtype = torch.bfloat16 if cfg.data.precision == "bf16-mixed" else torch.float16


    server = SwarmServer.create(
        start=False,
        initial_peers=cfg.network.initial_peers,
        host_maddrs=cfg.network.host_maddrs,
        announce_maddrs=cfg.network.announce_maddrs,
        listen_on=listen_on,
        announce_endpoint=announce_endpoint,
        device=cfg.device,
        update_period=int(cfg.network.expert_dht_update_period),
        dht_expiration=int(cfg.network.expert_dht_expiration),
        expert_cls=expert_cls,
        expert_uids=[expert_uid],
        hidden_dim=hidden_dim,
        optim_cls=optim_cls,
        clip_grad_norm=cfg.diloco.max_grad_norm,
        optim_kwargs=optim_kwargs,
        min_batch_size=1,
        max_batch_size=cfg.diloco.batch_size_per_step,
        fp16=cfg.data.precision in ("fp16-mixed", "bf16-mixed"),
        autocast_dtype=autocast_dtype,
        quant_config=cfg.quant if not cfg.disable_quant else None,
        dht=dht,
        cfg=cfg,
        stage_index=stage_index,
        checkpoint_dir=cfg.checkpoint_dir if cfg.checkpoint_dir else None,
        checkpoint_keep_history=cfg.checkpoint_keep_history,
        checkpoint_update_period=cfg.checkpoint_update_period,
        compression=compression["grad_compression"],
        skip_load_from_peers=cfg.network.skip_load_from_peers,
        skip_load_checkpoints=cfg.network.skip_load_checkpoints,
    )

    # Store batch_size_per_step in DHT for this expert
    from distqat.distributed.server.dht_handler import store_expert_batch_size, store_expert_inner_steps
    batch_size = cfg.diloco.batch_size_per_step
    inner_steps = cfg.diloco.inner_steps
    store_expert_batch_size(dht, expert_uid, batch_size, expiration=60, wait=True)
    store_expert_inner_steps(dht, expert_uid, inner_steps, expiration=60, wait=True)

    logger.info(f"Server created for expert {expert_uid} with batch_size_per_step={batch_size} and inner_steps={inner_steps}")

    # Start trainer if it's a head node
    if stage_index == 0:
        if trainer_in_process:
            start_trainer_inprocess(cfg=cfg, expert_index=expert_index, dht=dht, local_expert_backends=server.experts)
        else:
            start_trainer(cfg, expert_index)

    return server, dht, expert_uid


def main(
    cfg: Config,
    stage_index: Optional[int] = None,
    expert_index: Optional[int] = None,
    listen_on: str = "0.0.0.0:*",
    announce_endpoint: Optional[str] = None,
    trainer_in_process: bool = False,
):
    restart_event = threading.Event()
    restarted = False
    sleep_time = 0
    
    while True:
        # reset stage_index and expert_index to None to force auto-discovery on restart
        if restarted:
            stage_index, expert_index = None, None
        server, dht, expert_uid = start_server(
            cfg,
            stage_index,
            expert_index,
            listen_on,
            announce_endpoint,
            trainer_in_process=trainer_in_process,
        )
        shutdown_thread = threading.Thread(target=server.runtime.shutdown)
       
        def shutdown_callback():
            """Callback function to trigger graceful shutdown and restart the server"""
            nonlocal sleep_time
            restart_event.set()
            reassignment_monitor.stop()
            # remove_expert_from_dht(dht, expert_uid)

            expert_info = dht.get(expert_uid, return_future=False)

            expiration_time = None
            if expert_info is not None and hasattr(expert_info, "expiration_time"):
                expiration_time = getattr(expert_info, "expiration_time")
                sleep_time = 2*max(0, expiration_time - get_dht_time())

            shutdown_thread.start()

        reassignment_monitor = ReassignmentMonitorThread(
            expert_uid=expert_uid, dht=dht, check_period=10,
            shutdown_callback=shutdown_callback, daemon=True,
        )
        reassignment_monitor.start()
        signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            server.run()
        except KeyboardInterrupt:
            pass
        finally:
            reassignment_monitor.stop()
            reassignment_monitor.join(timeout=5)
            if shutdown_thread.is_alive():
                shutdown_thread.join(timeout=5)

            if cfg.wandb_project is not None:
                try:
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Failed to finish wandb for server: {e}")

        if not restart_event.is_set():
            break
        restart_event.clear()
        
        restarted = True
        sleep_time = random.uniform(sleep_time, sleep_time + 10)
        logger.info(f"Waiting {sleep_time} seconds until restarting server so that previous expert uid entry expires")
        time.sleep(sleep_time)

if __name__ == "__main__":
    import click
    
    parse_args_with_extra_kwargs = click.option("--stage-index", type=int, default=None)(parse_args)
    parse_args_with_extra_kwargs = click.option("--expert-index", type=int, default=None)(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option("--listen-on", type=str, default="0.0.0.0:*")(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option("--announce-endpoint", type=str, default=None)(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option("--trainer-in-process", is_flag=True)(parse_args_with_extra_kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*is used more than once. Remove its duplicate as parameters should be unique.*")
        res = parse_args_with_extra_kwargs(standalone_mode=False)
        if isinstance(res, int):
            quit() # Help has been called
        elif isinstance(res, tuple):
            cfg, extra_kwargs = res
            main(cfg, **extra_kwargs)
        else:
            raise ValueError(f"Unexpected return type: {type(res)}")