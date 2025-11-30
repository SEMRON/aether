# Distqat

> Decentralized, pipeline-parallel training with automatic expert discovery, DiLoCo-style optimization, and resilient failover.


## 🚀 Quick Start
**New to DistQat?** Check out our [Quick Start Guide](./docs/QUICK_START.md) to get up and running in 5 minutes!

For more information continue reading below or refer to the full documentation:
> 📊 **[Full documentation with Diagrams→](docs/README.md)**



## Table of Contents
- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
  - [System Roles](#system-roles)
  - [Lifecycle](#lifecycle)
- [Run Modes](#run-modes)
  - [Fully Local Sandbox](#fully-local-sandbox)
  - [Bring Your Own Machines](#bring-your-own-machines)
- [Configuration](#configuration)
- [Observability](#observability)
- [Resilience & Failover](#resilience--failover)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Roadmap & Inspirations](#roadmap--inspirations)

## Overview
Distqat is a collaborative training framework for large models. Each participant contributes GPU/ CPU capacity by hosting model shards that are chained into full pipelines on demand. Trainers stream batches from a shared-memory data server, dispatch forward/backward calls across the swarm, and use DiLoCo’s inner/outer optimization to synchronize the replicas with less communication overhead.

The design builds on lessons from projects such as OpenDiLoCo and SWARM parallelism, while focusing on:
- pipeline-parallel experts that can churn in and out,
- dynamic discovery through a libp2p-backed Distributed Hash Table (DHT),
- optional quantization
- proactive failover

## Key Capabilities
- **Pipeline-aware experts** – `src/distqat/distributed/server.py` hosts stages advertised in the DHT, letting clients assemble full end-to-end executors.
- **Adaptive trainers** – `src/distqat/distributed/client.py` spawns one trainer per complete pipeline (`trainer.py`), supervises health, and restarts crashed peers.
- **DiLoCo optimizer** – `src/distqat/distributed/optim/diloco.py` implements local inner updates with periodic global averaging, giving robustness on unreliable links.
- **Shared-memory data plane** – `src/distqat/distributed/data_server.py` shares batches via `multiprocessing.shared_memory`, so trainers avoid redundant preprocessing.
- **Quantization hooks** – `src/distqat/quant` exposes per-stage overrides to keep traffic manageable on commodity uplinks.
- **Param mirroring** – `src/distqat/distributed/param_mirror.py` keeps a CPU copy of expert weights and checkpoints for quick warm starts.
- **W&B-first monitoring** – `src/distqat/distributed/monitor.py` pushes fleet-wide metrics and pipeline health to a single run dashboard.


## Architecture

### System Roles
| Role | Binary | Description |
|------|--------|-------------|
| **Monitor** | `monitor.py` | Seeds the DHT, records active multiaddresses, streams DiLoCo progress/metrics to Weights & Biases, and writes `initial_peers.txt` for bootstrap. |
| **Client** | `client.py` | Watches the DHT for complete pipelines, spawns/stops trainers, controls the shared data server, and triggers expert reassignment when gaps appear. |
| **Data Server** | `data_server.py` | Preloads datasets defined in `Config.data`, shuffles with `BufferedShuffleIterable`, and serves batches via shared memory pools. |
| **Trainer** | `trainer.py` | Runs DiLoCo inner loops, performs forward/backward via `SwarmModel`, and logs metrics back to the DHT (`LocalMetrics`). |
| **Server / Expert** | `server.py` | Hosts a model shard (e.g., `resnet.tail`) with optional quantization, advertises itself to the DHT, and reacts to reassignment signals. |
| **Param Mirror (optional)** | `param_mirror.py` | Periodically pulls weights/optimizer state from peers for checkpointing and crash recovery. |

### Lifecycle
1. **Monitor boots** and starts a DHT node, recording its visible multiaddresses to `logs/<experiment>/initial_peers.txt`.
2. **Client connects** to the same DHT, launches the shared-memory data server, and waits for complete pipelines.
3. **Experts register** themselves (`SwarmServer.create`) with stage/expert UIDs. Auto-discovery back-fills gaps if a stage is missing.
4. **Client assembles pipelines** from the DHT view, spawns one trainer per complete pipeline, and restarts trainers that exit unexpectedly.
5. **Trainer loops** execute `inner_steps` local batches, participate in DiLoCo averaging rounds (`optimizers.get_diloco_optimizer_cls_kwargs`), and publish metrics.
6. **Failover loop** detects incomplete pipelines. `ReassignmentMonitorThread` on each server listens for reassignment signals and gracefully restarts into the swarm.

## Run Modes

### Fully Local Sandbox
Best for iterating on configs or operations demos.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python=3.10
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
source .venv/bin/activate
uv pip install --editable .

wandb login
python run_local.py  # defaults to configs/resnet18.yaml
```

`run_local.py` orchestrates the full stack (monitor → client → data server → baseline trainer → experts) and streams logs to `logs/<experiment_prefix>`.

### Bring Your Own Machines
1. **Trainer + monitor host**
   ```bash
   export PUBLIC_IP=<trainer_public_ip>
   wandb login
   python start_trainer_client.py --public-ip "${PUBLIC_IP}"
   ```
   Copy the peer addresses written to `logs/<experiment_prefix>/initial_peers.txt`.

2. **Server hosts**
   ```bash
   export INITIAL_PEERS='/ip4/<trainer_public_ip>/tcp/50000/p2p/<peer_id>'
   python start_servers.py \
     --public-ip "<this_server_ip>" \
     --num-servers 1 \
     --initial-peers "${INITIAL_PEERS}"
   ```
   Repeat on as many machines as you like; each process will auto-assign to open pipeline slots.

### Preview of adaptive batch sizing for heterogenous training
Here is an example of how heterogenous training will look like with adaptive batchsizing. Currently the batch size still has to be set manually.
1. **Trainer + monitor host**
   ```bash
   export PUBLIC_IP=<trainer_public_ip>
   wandb login
   python start_trainer_client.py --public-ip "${PUBLIC_IP}"
   ```
   Copy the peer addresses written to `logs/<experiment_prefix>/initial_peers.txt`.

2. **Server host with GPU**
   ```bash
   export INITIAL_PEERS='/ip4/<trainer_public_ip>/tcp/50000/p2p/<peer_id>'
   python start_servers.py \
     --public-ip "<this_server_ip>" \
     --num-servers 1 \
     --initial-peers "${INITIAL_PEERS}" \
     --config-path "configs/resnet18.yaml"
   ```

  3. **Server host with a CPU**
   ```bash
   export INITIAL_PEERS='/ip4/<trainer_public_ip>/tcp/50000/p2p/<peer_id>'
   python start_servers.py \
     --public-ip "<this_server_ip>" \
     --num-servers 1 \
     --initial-peers "${INITIAL_PEERS}" \
     --config-path "configs/resnet18.yaml" \
     --diloco-batch-size-per-step 1 \
     --device cpu
   ```

   If we look at the logs we should be able to see that the servers train at a similar speed since the performances are aligned with the batch size and thus they can use DiLoCo to average without the bottleneck of needing to wait for slower peers.

## Configuration
- **Configs** live in `configs/*.yaml`. The default `experiment_prefix`, DiLoCo parameters, and model pipeline (e.g. `resnet`, `gpt-neo`, `wav2vec`) live here.
- **Pydantic models** in `src/distqat/config.py` enforce structure and expose every field via CLI flags.
- **Quantization** – adjust `quant` with a `QuantConfiguration` disable with `--disable-quant`
- **Datasets** – update `DataConfig` to select CV/LLM/Speech tasks; the data server will automatically pick the right `collate_fn`.
- **Param mirroring** – enable/disable checkpoints via `param_mirror.enable` and tune `refresh_every`/`checkpoint_dir`.

To inspect the resolved config at runtime log `cfg.model_dump()` from custom scripts.

## Observability
- All components log to `logs/<experiment_prefix>/` with one file per role (e.g. `server_back_0.log`) when started with `run_local.py` otherwise with `0 to num_servers` when started with `start_servers.py` (e.g. `server_0.log`).
- `monitor.py` streams fleet metrics (loss, SPS, alive peers) to your configured W&B project.
- Trainers publish per-step telemetry to the DHT (`experiment_prefix_metrics`), consumable by automation or custom dashboards.

## Resilience & Failover
- `ReassignmentMonitorThread` (within each server) watches for incomplete pipelines and triggers a controlled shutdown so the server can grab a new slot.
- Run `python test_failover.py` to simulate staged shutdowns and validate that trainers keep progressing.
- Parameter mirroring ensures fresh experts can  `load_state_from_peers()` instead of random initialization if no server peers are available.

## Troubleshooting & FAQ
- **“Address already in use” when restarting locally** – stale processes may still be running. Try `pkill -f distqat` to clean up orphaned processes, then relaunch.
- **Trainers stall at “Auto-discovering pipeline gaps”** – ensure at least one expert per stage is reachable and that the monitor host’s port (default 50000) is open to inbound traffic.
- **Timeouts during averaging** – increase `diloco.averaging_timeout` or reduce batch size (`diloco.batch_size_per_step`) for slower links.
- **No metrics in W&B** – confirm `wandb login` succeeded everywhere and that the monitor is launched with `--wandb-run-id` when orchestrating multiple processes.

- **How do I run a baseline without the swarm?** – launch `start_trainer_client.py` to start the data server and then `trainer.py --run-locally --trainer-id "-1" --config-path "configs/resnet18.yaml" --network-initial-peers "[\"/ip4/127.0.0.1/tcp/50000/p2p/<peer-id>\"]"` to run a fully local model.
