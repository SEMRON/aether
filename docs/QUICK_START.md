# Quick Start Guide

Get up and running with DistQat in 5 minutes!

## Prerequisites

- Python 3.10+
- Linux OS (Most extensively tested on Ubuntu 22.04 and 24.04 but should work on others as well)
- If using a CUDA- or ROCM-capable GPU you need to install the CUDA or ROCM drivers first
- Allow TCP Ports to be open and accept incoming connection from outside (Ports: 50000-60000 for simplicity or otherwise 50000, 51000, 50500, 51500 are the minimum ports that should be open)
- Optional but highly recommended (especially for the GUI): A WandB account
- Huggingface Token that has access rights to ImageNet1k (https://huggingface.co/datasets/ILSVRC/imagenet-1k) if you want to train on it (eg. ResNet50 config)

## 1. Installation (On every machine you want to use)

Connect to your machine that you want to use for the distributed training and then install the framework doing the following.

```bash
# Clone repository
git clone https://github.com/SEMRON/aether.git distqat
cd distqat

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the shell again or start a new shell to make uv available
source $HOME/.local/bin/env

# Create environment
uv venv --python=3.10
source .venv/bin/activate

# If on a CUDA machine
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# If on a ROCM capable AMD machine
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# If on a CPU
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install distqat
uv pip install -e .

wandb login

```


## 2. Test installation by running Local Training
To test if the installation and everything worked correctly you can try a local training run with the following command. If there are any issues you can refer to [Common Issues](#common-issues) which might be helpful.

```bash
# if using a CPU only machine otherwise --device cpu can be ommitted or specified as cuda, otherwise it uses the value specified in the config file:
python run_local.py --config-path configs/resnet18.yaml --device cpu

# if not using WandB append the flag --wandb-project None: 
python run_local.py --config-path configs/resnet18.yaml --wandb-project None

# for pipeline parallel test
python run_local.py --config-path configs/resnet18_split.yaml --num-servers 2
```

After waiting a few seconds up to a minute if you see console logs that are similar to this:
```
Nov 26 17:29:58.382 [INFO] Complete pipeline found for expert_index 0: ['head', 'tail']
Nov 26 17:29:58.382 [INFO] Complete pipeline found for expert_index 1: ['head', 'tail']
Nov 26 17:29:58.382 [INFO] Found 2 complete and 0 incomplete pipelines
Nov 26 17:29:58.383 [INFO] [Step 0]
  Losses:
    distributed: n/a
    baseline:    2.37161
  Pipelines:
    Pipeline #0:
      - head: 0 peer(s)
          (no active peers)
      - tail: 0 peer(s)
          (no active peers)
    Pipeline #1:
      - head: 0 peer(s)
          (no active peers)
      - tail: 0 peer(s)
          (no active peers)
```

it means the installation and local training is working as expected and you can cancel the Training with `ctrl+c`.

## 3. Distributed Training using the Command Line

### On Machine 1 (Coordinator):

```bash
python start_trainer_client.py

# Wait a few seconds, then get the peer address from the logs or by checking:
cat logs/resnet18/initial_peers.txt
```

### On Machine 2+ (Workers):

```bash
export INITIAL_PEERS="<peer_address_from_machine_1>"

python start_servers.py \
    --initial-peers "${INITIAL_PEERS}"
```

Now after some time passes you should see output similar to the `local_run` on the coordinator node. If not something might have gone wrong so it would be a good idea to check the log files in `logs/*` on the different machines and refer to [Common Issues](#common-issues) or look into the full documentation. 

## 4. Examples
The `resnet18` config should work easily on most machines and does not have much requirements on the hardware. If you want to test out the pipeline parallelism on a small model, then use `resnet18_split.yaml`. This configuration splits the Resnet18 model into two stages. So you need at least 2 servers to train the model. 

## 5. Advanced 
Now if everything works as expected you can dive deeper into the distributed Training. You can test out different models with the GUI based on the provided config files in `configs/*.yaml` or write your own config file. You could also try to implement your own model, although depending on the model complexity that could be more or less challenging. Refer to [Advanced Topics in the README](./README.md#advanced-topics)

## Common Issues

### Leftover Processes
Sometimes when scripts are started and then stopped multiple times, it can lead to errors (eg. the dataserver fails because there is an old dataserver process running). 
`RuntimeError: Timed out waiting for data server manager to start`.
Run the following command and then try running the script again. This is the equvialent of "turn it off and on again".

```bash
pkill -f distqat
```

### Port in Use
```bash
kill -9 $(lsof -t -i:50000)
```

### CUDA Out-of-memory
Edit `configs/resnet18.yaml`:
```yaml
diloco:
  batch_size_per_step: 16  # Reduce from 64
```

### GRPC Connection refused error
Connection refused, check if port is open and accessible from outside. Will be shown in trainer logs.
If there is port mapping then specify the external port by setting `network-server-base-grpc-announceport` when run from the command line or setting the GRPC port for that server in the GUI. It's the port that's mapped to `51500` by default.

### Failed to connect to bootstrap peer
Check if the IP address of the initial peers is correct and the port is accessible. If port mapping is used then after copying the initial peers you need to change the port so from:
e.g.: `/ip4/213.173.111.105/tcp/50000/p2p/QmPkcZiABVfu41yA3qTF1LpmB9z2ZqjuwjJi2TeBj6fZd6` to `/ip4/213.173.111.105/tcp/<mapped port>/p2p/QmPkcZiABVfu41yA3qTF1LpmB9z2ZqjuwjJi2TeBj6fZd6`

### Hugging face authentication error
`datasets.exceptions.DatasetNotFoundError: Dataset 'ILSVRC/imagenet-1k' is a gated dataset on the Hub. You must be authenticated to access it.`
If you get this error it means the dataset you're trying to use is gated and you need to log in to hugging face either by calling `hf auth login` in the terminal or passing `HF_TOKEN` as an environment variable to your machine.

### Installation SSL: Certificate error 
```
python -m pip install --upgrade certifi \
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

### Protobuf decoding error 
```
Dec 31 11:11:48.344 [ERROR] [distqat.distributed.client.balanced_expert.forward:160] Tried to call forward for expert RemoteExpert(uid=full.0.105.0, endpoint=127.0.0.1:51500):
Traceback (most recent call last):
  File "/home/pahrendt/distqat/src/distqat/distributed/client/balanced_expert.py", line 147, in forward
    outputs = chosen_expert.stub.forward(forward_request, timeout=forward_timeout)
  File "/home/pahrendt/distqat/.venv/lib/python3.10/site-packages/grpc/_channel.py", line 1166, in __call__
    return _end_unary_response_blocking(state, call, False, None)
  File "/home/pahrendt/distqat/.venv/lib/python3.10/site-packages/grpc/_channel.py", line 996, in _end_unary_response_blocking
    raise _InactiveRpcError(state)  # pytype: disable=not-instantiable
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
	status = StatusCode.UNKNOWN
	details = "Unexpected <class 'google.protobuf.message.DecodeError'>: Error parsing message with type 'Tensor'"
	debug_error_string = "UNKNOWN:Error received from peer ipv4:127.0.0.1:51500 {grpc_status:2, grpc_message:"Unexpected <class \'google.protobuf.message.DecodeError\'>: Error parsing message with type \'Tensor\'"}"
>
```

In this case the batch size or gradient accumulation step size is too high causing the tensors which are send through RPC being too large. Reduce batch size or gradient accumulation step size until it works.

## Next Steps

- Read the [full README](README.md) for detailed documentation
- Check [configuration guide](README.md#configuration) to customize training
- See [examples](README.md#examples) for different models
- Review [troubleshooting](README.md#troubleshooting-faq) for common issues

## File Structure

```
distqat/
├── configs/                 # YAML configuration files
│   ├── resnet18.yaml       # Default config
│   ├── distilgpt2.yaml     # Language model config
│   └── ...
├── src/distqat/
│   ├── distributed/
│   │   ├── monitor.py      # DHT monitoring
│   │   ├── client.py       # Trainer orchestration
│   │   ├── server.py       # Expert hosting
│   │   └── trainer.py      # Training execution
│   └── models/             # Model definitions
├── start_trainer_client.py # Launch coordinator
├── start_servers.py        # Launch workers
└── logs/                   # Training logs (created at runtime)
```

## Monitoring Training

### Check Logs
```bash
# Monitor logs
tail -f logs/resnet18/client.log
tail -f logs/resnet18/server_*.log

# Watch WandB dashboard (if configured)
# Visit https://wandb.ai/your-entity/distqat
```

### Verify Training is Running
Look for these messages in client logs:

```
INFO: Found expert head.0.0.0 at 0.0.0.0:65297
INFO: Complete pipelines: 2 - [0, 1]
INFO:  Spawned trainer 0 with PID 2508688
INFO: Trainer 0 Step #100 loss = 2.34567 sps = 12.5
```

## Stop Training

```bash
# Ctrl+C in the terminal running the process

# Or kill all processes
pkill -f distqat

# Clean up ports if needed
kill -9 $(lsof -t -i:50000)
kill -9 $(lsof -t -i:50500)
kill -9 $(lsof -t -i:51000)
kill -9 $(lsof -t -i:52555)
```

## Test Failover

```bash
python test_failover.py --public-ip ${PUBLIC_IP}
```

This script automatically tests the failover mechanism by killing a server and verifying recovery.

---

**Need Help?** See the [Troubleshooting & FAQ](README.md#troubleshooting-faq) section in the main README.
