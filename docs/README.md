# Documentation

<div align="center">

**Distributed Quantization-Aware Training Framework**

A decentralized framework for training large models with model parallelism, automatic failover, and communication-efficient optimization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


</div>

---

## 📑 Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [System Components](#system-components)
  - [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Local Training](#local-training)
  - [Distributed Training](#distributed-training)
  - [Cloud Deployment with SkyPilot](#cloud-deployment-with-skypilot)
- [Examples](#examples)
- [Troubleshooting & FAQ](#troubleshooting-faq)
- [Advanced Topics](#advanced-topics)
- [Citations](#citations)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

(quick-start)=
## Quick Start

**New to DistQat?** Check out our [Quick Start Guide](QUICK_START.md) to get up and running in 5 minutes!

For detailed documentation, continue reading below.

---

(overview)=
## Overview

**DistQat** (Distributed Quantization-Aware Training) is a framework for training neural networks across distributed nodes using model parallelism and quantization-aware techniques. Built on top of [Hivemind](https://github.com/learning-at-home/hivemind), DistQat enables:

- **Model Parallel Training**: Split large models across multiple nodes with pipeline parallelism
- **Automatic Node Discovery**: Dynamically discover and connect to available compute resources
- **Fault Tolerance**: Automatic failover when nodes disconnect or fail
- **Communication Efficiency**: Uses DiLoCo (Distributed Low-Communication) optimization to minimize network overhead
- **Quantization Support**: Train with quantization-aware techniques

This framework is particularly useful for:
- Training models that don't fit on a single GPU
- Utilizing heterogeneous compute resources across the internet
- Experimenting with distributed training protocols
- Research in decentralized AI training

---

(key-features)=
## Key Features

### Distributed Architecture
- **Swarm-based P2P Network**: Nodes communicate via DHT (Distributed Hash Table) for peer discovery
- **Dynamic Node Joining/Leaving**: Add or remove compute nodes without stopping training
- **Automatic Pipeline Discovery**: System automatically discovers available pipeline stages and forms complete training paths

### Fault Tolerance
- **Automatic Failover**: When a node fails, the system automatically reassigns work to healthy nodes if possible
- **Checkpoint & Resume**: Periodic model checkpointing training resumption
- **Graceful Degradation**: Training continues even when some nodes are unavailable

### Communication Efficiency
- **DiLoCo Optimization**: Decoupled inner and outer optimizers minimize synchronization overhead
- **Configurable Sync Intervals**: Tune communication frequency vs. convergence speed tradeoffs

### Monitoring & Observability
- **Wandb Integration**: Real-time metrics logging and visualization
- **Per-Node Metrics**: Track performance of individual compute nodes
- **DHT State Monitoring**: Observe network topology and node health

---

(architecture)=
## Architecture

(system-components)=
### System Components

DistQat consists of four main components that work together to enable distributed training:

```
┌─────────────────────────────────────────────────────────────┐
│                        DHT Network                          │
│              (Distributed Hash Table for P2P)               │
└─────────────────────────────────────────────────────────────┘
         ▲              ▲              ▲              ▲
         │              │              │              │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │ Monitor │    │ Client  │    │ Server  │    │ Server  │
    │         │    │         │    │(Stage 0)│    │(Stage 1)│
    └─────────┘    └────┬────┘    └─────────┘    └─────────┘
                        │
                   ┌────┴────┐
                   │ Trainer │
                   │ (spawns │
                   │dynamical│
                   │   ly)   │
                   └─────────┘
```

> 📊 **[Detailed Architecture Diagram →](./diagrams/ARCHITECTURE.md)**  
> See the full architecture diagram with Mermaid code and detailed component breakdown.

#### 1. **Monitor** (`src/distqat/distributed/monitor.py`)

The monitor is the entry point that initializes the DHT network and tracks system state.

**Responsibilities:**
- Create and maintain the DHT node
- Store and share initial peer addresses for other nodes to connect
- Collect and aggregate metrics from trainers
- Log metrics to Wandb for visualization

**Key Parameters:**
- `refresh_period`: How often to poll for metrics (default: 300s)
- `store_ip_addresses_path`: File path to save initial peer addresses

#### 2. **Server/Expert** (`src/distqat/distributed/server/`)

Servers host individual stages of the model pipeline (e.g., "head" and "tail" of a ResNet).

**Responsibilities:**
- Host a specific pipeline stage (expert)
- Register expert UID in the DHT for discovery
- Execute forward and backward passes for assigned stage
- Participate in DiLoCo parameter synchronization
- Maintain checkpoints and model state
- Handle automatic reassignment when nodes fail

**Key Concepts:**
- **Expert UID**: Format `{stage}.0.{expert_index}.0` (e.g., "head.0.0.0")
- **Stage Index**: Which stage in the pipeline (0, 1, 2, ...)
- **Expert Index**: Which replica of this stage (for redundancy/parallelism)
- **Auto-discovery**: Servers can automatically find gaps in the pipeline to fill

#### 3. **Client** (`src/distqat/distributed/client.py`)

The client discovers available experts and dynamically spawns trainers.

**Responsibilities:**
- Poll DHT to discover available experts
- Find complete pipeline paths (all stages available)
- Spawn trainer processes for each complete path
- Load balance across available experts
- Monitor trainer health and restart if needed

**Key Features:**
- **Dynamic Trainer Spawning**: Automatically creates trainers when pipelines become available
- **Expert Balancing**: Distributes work across multiple replicas of the same stage
- **Health Monitoring**: Detects and recovers from trainer failures

#### 4. **Trainer** (`src/distqat/distributed/trainer.py`)

Trainers execute the actual training loop using remote experts.

**Responsibilities:**
- Load dataset and create data loaders
- Execute forward pass through remote experts
- Compute loss and execute backward pass
- Report metrics to the monitor

**Training Modes:**
- **Distributed Mode**: Uses remote experts via SwarmModel
- **Baseline Mode**: Uses local model computation via SwarmBaselineModel (for comparison)

(how-it-works)=
### How It Works

#### Initial Setup

1. **Monitor starts** and creates DHT, writes peer address to file
2. **Client reads** peer address and connects to DHT
3. **Data Server starts** Data server starts and streams data to shared memory
4. **Servers start**, connect to DHT, and register their expert UIDs
5. **Client discovers** available experts and forms complete pipelines
6. **Trainers spawn** dynamically for each complete pipeline

> 📊 **[Node Arrangement Diagram →](./diagrams/NODE_ARRANGEMENT.md)**  
> See detailed node arrangement, port assignments, and replica distribution patterns.

#### Training Flow

```
Trainer                    Server (Stage 0)         Server (Stage 1)
   │                              │                         │
   ├──── Forward (input) ────────>│                         │
   │                              ├──── Forward (hidden) ──>│
   │                              │                         │
   │<──────────────────────────── Output ───────────────────┤
   │                              │                         │
   ├──── Backward (grad) ──────────────────────────────────>│
   │                              │<──── Backward (grad) ───┤
   │<──── Backward (grad) ────────┤                         │
   │                              │                         │
   ├──── DiLoCo Sync (every N steps) ─────────────────────> │
                     (between servers of the same stage)
```

**Inner Loop (Local Optimization):**
1. Trainer sends batch to first server stage
2. Each stage processes and forwards to next stage
3. Final stage sends logits to Trainer
4. Trainer calculates loss and starts backward pass
5. Gradients flow backward through stages
6. Each server updates its local parameters
7. Repeat for `inner_steps` iterations

**Outer Loop (Global Synchronization):**
1. After `inner_steps`, servers of the same stage synchronize parameters
2. Uses outer optimizer (typically SGD with momentum)
3. Averages updates across all replicas of each stage
4. Updates global model state

> 📊 **[Training Flow Diagram →](./diagrams/TRAINING_FLOW.md)**  
> See detailed sequence diagrams and timing charts for forward/backward passes.

#### Failover Mechanism

When a server node fails:

```
Before Failure:
Trainer → Server A (head.0.0.0) → Server B (tail.0.0.0)

After Server B Fails:
Trainer → Server A (head.0.0.0) → [waiting for new tail stage]

After New Server C Joins:
Trainer → Server A (head.0.0.0) → Server C (tail.0.1.0)
```

> 📊 **[Failover Mechanism Diagram →](./diagrams/FAILOVER.md)**  
> See detailed state diagrams, timeline charts, and topology changes during failover.

**Failover Process:**
1. Trainer detects server unavailability (timeout)
2. Client marks expert as unavailable
3. Client searches for alternative experts in DHT
4. New server automatically discovers gap and fills it
5. New server loads latest checkpoint
6. Training resumes with new topology

**Automatic Gap Discovery:**
- Servers scan DHT for missing expert indices
- Automatically assign themselves to unfilled positions
- Load most recent checkpoint for that stage
- Register in DHT and begin accepting requests

---

(installation)=
## Installation

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **CUDA**: For GPU support (recommended)
- **Operating System**: Linux (tested on Ubuntu and Rocky Linux 8.10)

### Quick Start (Ubuntu)

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python=3.10
source .venv/bin/activate

# Install PyTorch with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Install DistQat
uv pip install .

# For development (editable install)
uv pip install --editable .
```

### Installing Hivemind Dependency

DistQat depends on Hivemind for P2P networking. You have two options:

#### Option A: Use Pre-compiled Binary (Easier)

```bash
git submodule update --init externals/hivemind
pip install -e externals/hivemind
```

#### Option B: Build from Source (Required for some systems)

This option requires Go 1.25+ for building `go-libp2p-daemon`:

```bash
# Initialize hivemind submodule
git submodule update --init externals/hivemind

# Install Go
wget https://go.dev/dl/go1.25.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.25.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Install hivemind requirements first
wget -O /tmp/hivemind_requirements.txt https://raw.githubusercontent.com/learning-at-home/hivemind/937198fd364d20edeafd1044b12c3b88521d7701/requirements.txt
uv pip install -r /tmp/hivemind_requirements.txt

# Build and install hivemind
source .venv/bin/activate
uv pip install pip
pip install --global-option=build_py --global-option="--buildgo" --no-use-pep517 \
  git+https://github.com/learning-at-home/hivemind.git@937198fd364d20edeafd1044b12c3b88521d7701
```

### Verify Installation

```bash
python -c "import distqat; import hivemind; print('DistQat installed successfully!')"
```

---

(configuration)=
## Configuration

DistQat uses YAML configuration files to define training parameters, model architecture, and network settings.

### Configuration File Structure

Example configuration (`configs/resnet18.yaml`):

```yaml
# WandB logging
wandb_project: "distqat"
experiment_prefix: "resnet18"

# Dataset configuration
data:
  dataset_name: "uoft-cs/cifar10"
  dataset_split: "train"
  task_type: "cv"
  num_workers: 0
  precision: "fp16-mixed"
  num_channels: 3
  img_size: 32

# DiLoCo optimization settings
diloco:
  inner_optim:
    type: "adam"
    adam_lr: 4e-4
    adam_weight_decay: 0.1
  outer_optim:
    type: "sgd"
    sgd_lr: 0.07
    sgd_momentum: 0.9
    sgd_nesterov: true
  inner_steps: 500      # Local optimization steps
  outer_steps: 100      # Global synchronization steps
  batch_size_per_step: 64
  averaging_timeout: 60.0

# Model pipeline stages
model_pipeline:
  pipeline:
    - model_name: "resnet18.head"
      num_classes: 10
    - model_name: "resnet18.tail"
      num_classes: 10
  forward_timeout: 90.0
  backward_timeout: 180.0

# Checkpointing
param_mirror:
  enable: true
  refresh_every: 30
  checkpoint_dir: "checkpoints/resnet18"

# Device settings
world_size: 2
device: "cuda"
max_expert_index: 128
```

### Configuration Sections Explained

#### Data Configuration
- `dataset_name`: HuggingFace dataset identifier
- `task_type`: ["cv", "llm" , "speech", "image_gen", "node_pred", "rl"]
- `precision`: Training precision ("fp16-mixed", "bf16", "fp32")

#### DiLoCo Configuration
- `inner_steps`: Number of local SGD steps before synchronization
- `outer_steps`: Total number of synchronization rounds
- `batch_size_per_step`: Batch size for each training step
- `inner_optim`: Local optimizer settings (Adam, AdamW, SGD)
- `outer_optim`: Global optimizer for parameter averaging (typically SGD with momentum)

#### Model Pipeline Configuration
- `pipeline`: List of model stages in order
  - Each stage defined by `model_name` (format: `{model}.{stage}`)
  - Available models: resnet18, resnet50, resnet101, distilgpt2, gptneo, wav2vec2
  - Available stages: head, body, tail, full

#### Network Configuration (CLI arguments)
- `initial_peers`: Comma-separated list of DHT peer addresses
- `host_maddrs`: Multiaddress for listening
- `announce_maddrs`: Multiaddress to announce to other peers

---

(usage)=
## Usage

(local-training)=
### Local Training

Train everything on a single machine (useful for testing):

```bash
# Ensure WandB is configured (or disable it in config)
wandb login

# Start training
python run_local.py
```

This will:
1. Start a monitor process
2. Start a client process that spawns trainers
3. Use local model computation (baseline mode)

Logs are saved to `logs/{experiment_prefix}/`:
- `monitor.log`: Monitor process logs
- `client.log`: Client discovery and trainer management logs
- `trainer_*.log`: Individual trainer logs

(distributed-training)=
### Distributed Training

Train across multiple machines using remote experts.

#### Step 1: Start Monitor & Client (Machine 1)

```bash
# Set your public IP address
export PUBLIC_IP="XX.XXX.XXX.XX"

# Start the orchestrator (monitor + client)
python start_trainer_client.py --public-ip ${PUBLIC_IP}
```

Monitor the logs to find the initial peer address:

```bash
tail -f logs/{experiment_prefix}/initial_peers.txt
```

You should see output like:
```
/ip4/XX.XXX.XXX.XX/tcp/50000/p2p/QmXAm92bj4biVVj6zHvj2ei5YiqqrW3brcVunYZ6HTipej,/ip4/127.0.0.1/tcp/50000/p2p/QmXAm92bj4biVVj6zHvj2ei5YiqqrW3brcVunYZ6HTipej
```

The first address is for remote connections, the second for local.

#### Step 2: Start Servers (Machine 2+)

On additional machines, start server processes:

```bash
# Set environment variables
export PUBLIC_IP="YY.YYY.YYY.YY"
export INITIAL_PEERS="/ip4/XX.XXX.XXX.XX/tcp/50000/p2p/QmXAm92bj4biVVj6zHvj2ei5YiqqrW3brcVunYZ6HTipej"

# Start multiple servers
python start_servers.py \
    --public-ip "${PUBLIC_IP}" \
    --num-servers 2 \
    --initial-peers "${INITIAL_PEERS}"
```

**Parameters:**
- `--num-servers`: Number of server instances to start on this machine
- `--public-ip`: Public IP address for remote connections
- `--initial-peers`: Peer address from Step 1 (comma-separated for multiple)

#### Step 3: Monitor Training

Watch the client logs to see trainers being spawned:

```bash
tail -f logs/{experiment_prefix}/client.log
```

View metrics in WandB dashboard (if configured).

(cloud-deployment-with-skypilot)=
### Cloud Deployment with SkyPilot

Deploy DistQat on cloud providers using [SkyPilot](https://skypilot.readthedocs.io/):

```bash
# Set WandB API key
export WANDB_API_KEY="your-wandb-key"

# Launch trainer + client (first run)
sky launch -c trainer-cluster ./skypilot/start_trainer_client.yaml \
    --secret WANDB_API_KEY

# Get the initial peer address from logs
# Then launch server workers
sky launch -c server-cluster ./skypilot/start_servers.yaml \
    --env NUM_SERVERS=8 \
    --env INITIAL_PEERS="/ip4/.../tcp/50000/p2p/..." \
    --secret WANDB_API_KEY

# For subsequent runs (skip setup)
sky launch -c trainer-cluster ./skypilot/start_trainer_client.yaml \
    --secret WANDB_API_KEY \
    --no-setup

sky launch -c server-cluster ./skypilot/start_servers.yaml \
    --env NUM_SERVERS=8 \
    --env INITIAL_PEERS="/ip4/.../tcp/50000/p2p/..." \
    --secret WANDB_API_KEY \
    --no-setup
```

---

(examples)=
## Examples

### Example 1: Training ResNet18 on CIFAR-10

```bash
# Use provided config
python run_local.py
# This uses configs/resnet18.yaml by default
```

### Example 2: Training DistilGPT2

```yaml
# configs/distilgpt2.yaml
data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"
  task_type: "nlp"

model_pipeline:
  pipeline:
    - model_name: "distilgpt2.head"
    - model_name: "distilgpt2.body"
    - model_name: "distilgpt2.tail"
```

```bash
python run_local.py --config-path configs/distilgpt2.yaml
```


This script:
1. Starts monitor, client, and multiple servers
2. After 20 seconds, kills one server
3. System automatically detects failure and reassigns work
4. New servers can be added dynamically

Watch the logs to observe failover in action:

```bash
tail -f logs/{experiment_prefix}/client.log
tail -f logs/{experiment_prefix}/server_*.log
```

### Example 3: Adding/Removing Nodes Dynamically

While training is running:

**Add a new server node:**
```bash
python start_servers.py \
    --public-ip "${PUBLIC_IP}" \
    --num-servers 1 \
    --initial-peers "${INITIAL_PEERS}"
```

**Remove a server node:**
Simply kill the server process (Ctrl+C or `pkill`). The system will automatically reassign work.

---

(troubleshooting-faq)=
## Troubleshooting & FAQ

### Common Issues

#### 1. Leftover Processes from Previous Run

**Symptom:** New run fails with "address already in use" or hanging connections

**Solution:**
```bash
# Kill all distqat processes
pkill -f distqat

# Or more specifically
pkill -f "python.*start_trainer_client.py"
pkill -f "python.*start_servers.py"
pkill -f "python.*monitor.py"

# Check if ports are still in use
lsof -i :50000  # Monitor port
lsof -i :50500  # Server port
lsof -i :51000  # Client port
lsof -i :52555  # Dataserver port

# If needed, kill processes by port
kill -9 $(lsof -t -i:50000)
```

#### 2. Servers Not Discovered

**Symptom:** Client doesn't spawn trainers, logs show "No complete pipelines found"

**Possible Causes:**
- Servers haven't finished initializing
- Network connectivity issues
- Incorrect initial peers configuration
- DHT not synchronized

**Solutions:**
```bash
# Wait longer (30-60 seconds) for DHT to synchronize
# Check server logs for errors
tail -f logs/{experiment_prefix}/server_*.log

# Verify servers registered successfully
grep "ReassignmentMonitorThread started for expert" logs/{experiment_prefix}/server_*.log

# Check client discovery logs
grep "Complete pipelines:" logs/{experiment_prefix}/client.log

# Verify network connectivity
ping <server_ip>
telnet <server_ip> 50500
```

#### 3. CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size in config
diloco:
  batch_size_per_step: 32  # Try smaller values: 16, 8, 4

# Enable mixed precision (if not already)
data:
  precision: "fp16-mixed"

# Use CPU for some servers (if needed)
python src/distqat/distributed/server.py --device cpu ...
```

#### 4. Slow Training / Timeouts

**Symptom:** Forward/backward passes timing out, slow samples per second

**Solutions:**
```yaml
# Increase timeouts in config
model_pipeline:
  forward_timeout: 180.0   # Increase from 90.0
  backward_timeout: 360.0  # Increase from 180.0

diloco:
  averaging_timeout: 120.0  # Increase from 60.0
```


#### 5. Wandb Authentication Errors

**Symptom:** "wandb: ERROR Failed to authenticate"

**Solution:**
```bash
# Login to WandB
wandb login
```

#### 6. Checkpoint Loading Failures

**Symptom:** "Failed to load checkpoint" or mismatched state dict

**Solutions:**
```bash
# Clear old checkpoints
rm -rf checkpoints/{experiment_prefix}/*

# Verify checkpoint directory permissions
ls -la checkpoints/{experiment_prefix}/

# Check disk space
df -h
```

### Frequently Asked Questions

#### Q: How many servers do I need for training?

**A:** Minimum is one server per pipeline stage. For example, if your pipeline has 2 stages (head, tail), you need at least 2 servers. Adding more servers (replicas) improves fault tolerance and convergence speed through data parallelism.

#### Q: Can I mix CPU and GPU nodes?

**A:** Yes! You can specify `--device cpu` or `--device cuda` when starting servers. However, note that CPU nodes will be much slower and may become bottlenecks. Averaging may fail.

#### Q: How does automatic failover work?

**A:** When a server fails:
1. Client detects missing stage for expert
2. Client stops that experts trainer
3. New servers automatically discover gaps via DHT
4. New server loads parameters from peers and registers
5. Client discovers new server and restarts the stopped trainer or spawns a new one

#### Q: What happens if the monitor dies?

**A:** The monitor is primarily for metrics collection. If it dies, training continues but metrics won't be logged to WandB. You can restart the monitor at any time.

#### Q: What happens if the client dies?

**A:** Trainers will stop since they're spawned by the client. Servers will remain running. Restart the client to resume training.

#### Q: Can I change the config during training?

**A:** Some parameters can be changed by restarting trainers/servers (batch size, learning rates), but structural changes (pipeline stages, model architecture) require a full restart.


#### Q: What's the difference between inner_steps and outer_steps?

**A:**
- `inner_steps`: Local optimization steps between synchronization (higher = less communication)
- `outer_steps`: Total number of synchronization rounds (outer_steps × inner_steps = total training steps)

#### Q: How do I debug connection issues?

**A:**
```bash
# Enable debug logging
export HIVEMIND_VERBOSITY=DEBUG

# Check DHT connectivity
python -c "
from hivemind import DHT
dht = DHT(start=True, initial_peers=['${INITIAL_PEERS}'])
print(dht.get_visible_maddrs())
"

# Verify firewall rules
sudo ufw status
sudo iptables -L

# Test port accessibility
nc -zv <server_ip> 50000
```

### Quick Fixes Reference

| Issue | Quick Fix |
|-------|-----------|
| Leftover processes | `pkill -f distqat` |
| Port in use | `kill -9 $(lsof -t -i:50000)` |
| OOM errors | Reduce `batch_size_per_step` in config |
| WandB errors | `wandb login` |
| DHT sync issues | Wait 30-60s, check `initial_peers` |
| Checkpoint errors | `rm -rf checkpoints/{exp_prefix}/*` |

---

(advanced-topics)=
## Advanced Topics

### Adaptive Batch Sizing for Heterogeneous Training

Preview adaptive batch sizing to keep heterogeneous trainers in sync while using DiLoCo averaging. Batch sizes still require manual tuning, but the workflow below shows how to stage GPU- and CPU-based servers so that throughput stays aligned. In the future this will be done automatically.

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

3. **Server host with CPU**

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

   Monitor the logs to confirm that both servers report similar step times, indicating that the batch sizes are balancing performance across heterogeneous hardware.

### Custom Models

To add a new model architecture:

1. Create model file in `src/distqat/models/`:

```python
# src/distqat/models/mymodel.py
import torch.nn as nn

def head_sample_input(batch_size, <parameters>):
    return ...

def tail_sample_input(batch_size, <parameters>):
    return ...

@register_expert_class("mymodel.full", head_sample_input)
class MyModelFull(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Full model definition
        
@register_expert_class("mymodel.head", head_sample_input)
class MyModelHead(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # First stage

@register_expert_class("mymodel.tail", tail_sample_input)
class MyModelTail(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Last stage
```

2. Register in `src/distqat/models/__init__.py`:

```python
from .mymodel import MyModelFull, MyModelHead, MyModelTail

MODEL_TYPES = {
    # ... existing models ...
    "mymodel.full": MyModelFull,
    "mymodel.head": MyModelHead,
    "mymodel.tail": MyModelTail,
}
```

3. Create config file `configs/mymodel.yaml` and run!

### Quantization

Disable quantization:
```bash
python start_servers.py --disable-quant ...
```

### Custom Optimizers

DistQat supports custom inner and outer optimizers:

```yaml
diloco:
  inner_optim:
    type: "adamw"
    adam_lr: 1e-4
    adam_weight_decay: 0.01
  outer_optim:
    type: "sgd"    # Typically SGD with momentum for outer loop
    sgd_lr: 0.1
    sgd_momentum: 0.9
    sgd_nesterov: true
```

### Parameter Mirroring & Checkpointing

Configure checkpoint frequency and location:

```yaml
param_mirror:
  enable: true
  refresh_every: 30 # save every 30 seconds
  checkpoint_dir: "checkpoints/myexp"
```

Checkpoints include:
- Model parameters for each stage
- Optimizer states (inner and outer)
- Training step counter
- Random states for reproducibility

---

(citations)=
## Citations

If you use DistQat in your research, please cite the frameworks it builds upon:

```bibtex
@inproceedings{ryabinin2021towards,
  title={Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts},
  author={Ryabinin, Max and Gusev, Anton},
  booktitle={NeurIPS 2020 Workshop on Pre-registration in Machine Learning},
  year={2020}
}
```

```bibtex
@inproceedings{ryabinin2023swarm,
  title={{SWARM} Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient},
  author={Ryabinin, Max and Dettmers, Tim and Diskin, Michael and Borzunov, Alexander},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={29416--29440},
  year={2023}
}
```


```bibtex
@misc{jaghouar2024opendiloco,
  title={OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training}, 
  author={Sami Jaghouar and Jack Min Ong and Johannes Hagemann},
  year={2024},
  eprint={2407.07852},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

(acknowledgements)=
## 🙏 Acknowledgements

DistQat is built upon the following open-source projects:

- **[Hivemind](https://github.com/learning-at-home/hivemind)**: Decentralized deep learning framework
- **[SWARM Parallelism](https://github.com/yandex-research/swarm)**: Communication-efficient training protocols
- **[OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo)**: Distributed Low-Communication optimization
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[WandB](https://wandb.ai/)**: Experiment tracking and visualization

Special thanks to the authors and maintainers of these projects for making decentralized AI training accessible.

---

(contact)=
## 📧 Contact

For questions, issues, or collaboration opportunities, please:
- Open an issue on GitHub
- Contact the maintainers

---

<div align="center">

**Made with ❤️ for decentralized AI**

</div>
