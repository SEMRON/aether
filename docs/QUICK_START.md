# Quick Start Guide

Get up and running with DistQat in 5 minutes!

## Prerequisites

- Python 3.10+
- Linux OS (Tested on Ubuntu 24.04, should work on others as well)
- If using a CUDA- or ROCM-capable GPU you need to install the CUDA or ROCM drivers first
- Optional but highly recommended (especially for the GUI): A WandB account

## 1. Installation (On every machine you want to use)

Connect to your machine that you want to use for the distributed training and then install the framework doing the following.

```bash
# Clone repository
git clone https://github.com/SEMRON/distqat.git

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the shell again or start a new shell to make uv available

# Create environment
cd distqat
uv venv --python=3.10
source .venv/bin/activate

# Install using make, default should work on most machines CPU and Nvidia CUDA
make install

# or for AMD ROCM
# make install-rocm

# or you can also adapt the Makefile to change the correct index file 

# or simply run 
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
# uv pip install -e .

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

## 4. Distributed Training using the Command Line

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

## 4. Using the GUI for Distributed Training
The GUI is still experimental so some unexpected bugs may arise. However it creates a unified interface that allows you to see all the logs from the different machines in one place, remotely install the framework on your different machines and in general can be helpful to get a better overview of the workflow.

Start the GUI on your local machine/ a machine not used for training since it works by SSH'ing into the machines used for training and cannot currently "SSH into itself". This machine needs to be able to access the other machines you want to use for training via SSH. If the other machines are secured through private/ public keys, then you must specify the path to that key file in the GUI when adding one of your servers.

```bash
python run_gui.py [--port <your_preferred_port, 8080 by default>]
```
Also check the std output in the terminal of the GUI process, since some errors might only get shown there. In general however, errors should be shown in the GUI itself.

If running it on a remote machine, you might need to open port forwarding to be able to view the GUI on your local browser:
```bash
# 8080 port is set by default, otherwise specify your preferred port you used when running the GUI
ssh -L 8080:localhost:8080 root@xx.xx.xx.xx
```

In the Server tab of the GUI you can add the data of the machines you want to use for the distributed Training. The machines either need to already have the framework installed as described in [Installation](#1-installation-on-every-machine-you-want-to-use-) or can be remotely installed by pressing the `Create Server Setup` button after selecting a server and then following the instructions, i.e. clicking `Generate` and then `Run Setup Script on Server`. Check the output in the GUI and if no error occured, then it should be correctly installed on the machine.

In the Orchestrator Tab you can then setup and start the distributed Training.
First you need to start a Head node on one of your machines, this will be the initial peer the worker servers will connect to. At the same time it will also start the monitor which will log the training progress. For this to work seemlessly, specify your WandB API key in the specified field. 

After having started the head node and received the initial peer address, you can start your servers to start the training process. During a training process you can always add a new server and join the ongoing training process.

After starting the training you can switch to the Monitor tab to keep track of your training progress and do some error checking in case anything goes wrong. The log files of the different processes and machines are gathered on WandB and downloaded to the GUI so that the can be live inspected. The Error watcher regularly scans the process and server logs for any errors for a quick overview. Note that it is just a simple error watcher and errors shown here do not necessarily mean, the Training is failing but it can still be a good first insight if something is not working as expected.

Currently, the training stops if the GUI is reloaded.

**Note:** You need to have some ports open for the distributed training to work. For more information on which ports to be open, see [Port Assignment Pattern](diagrams/NODE_ARRANGEMENT.md#port-assignment-pattern).
If the port is open but it is mapped to a different external port you can specify that port by editing the GRPC field for that server in the orchestrator.


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
