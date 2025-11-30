# Index

Complete guide to all DistQat documentation resources.

## 📚 Main Documentation

### Getting Started

- **[README](README.md)** - Main documentation with comprehensive guides
  - Overview and features
  - Installation instructions
  - Architecture explanation
  - Usage examples
  - Troubleshooting & FAQ
  - Advanced topics

- **[QUICK_START](QUICK_START.md)** - Get running in 5 minutes
  - Fast installation guide
  - Local training quickstart
  - Distributed training quickstart
  - Common issues and fixes

### Development

- **CONTRIBUTING.md** - Contribution guidelines (see repository root)
  - Development setup
  - Code style and standards
  - Testing guidelines
  - Pull request process
  - Adding new models

## 🖼️ Architecture Diagrams

Visual guides to understanding DistQat's architecture and operation.

### Component Diagrams

- **[Architecture Overview](diagrams/ARCHITECTURE.md)**
  - System component diagram
  - DHT network topology
  - Component interactions
  - Data flow visualization

- **[Node Arrangement](diagrams/NODE_ARRANGEMENT.md)**
  - Multi-machine deployment layout
  - Port assignment patterns
  - Expert UID namespace
  - Replication and load balancing

### Process Diagrams

- **[Training Flow](diagrams/TRAINING_FLOW.md)**
  - Forward/backward pass sequences
  - DiLoCo synchronization
  - Inner vs outer optimization loops
  - Timing and communication patterns

- **[Failover Mechanism](diagrams/FAILOVER.md)**
  - Failure detection process
  - Automatic recovery timeline
  - Node state transitions
  - Topology changes during failover

## 📖 Configuration

### Configuration Files

Located in `configs/` directory:

- `resnet18.yaml` - ResNet18 on CIFAR-10 (default)
- `resnet50.yaml` - ResNet50 configuration
- `resnet101.yaml` - ResNet101 configuration
- `distilgpt2.yaml` - DistilGPT2 language model
- `distilgpt2_full.yaml` - Full DistilGPT2 model
- `gptneo.yaml` - GPT-Neo configuration
- `gptneo_full.yaml` - Full GPT-Neo model
- `wav2vec2.yaml` - Wav2Vec2 audio model
- `wav2vec2_full.yaml` - Full Wav2Vec2 model

### Configuration Sections

Each config file contains:
- **Data**: Dataset and preprocessing settings
- **DiLoCo**: Optimization parameters (inner/outer optimizers)
- **Model Pipeline**: Pipeline stage definitions
- **Param Mirror**: Checkpointing configuration
- **Data Server**: IPC settings for data sharing
- **Device**: Hardware configuration

## 🔧 API Reference

### Core Modules

#### Distributed Components

- **`distqat.distributed.monitor`** - DHT monitoring and metrics aggregation
- **`distqat.distributed.client`** - Expert discovery and trainer orchestration
- **`distqat.distributed.server`** - Expert hosting and request handling
- **`distqat.distributed.trainer`** - Training loop execution

#### Models

- **`distqat.models.resnet`** - ResNet variants (18, 50, 101)
- **`distqat.models.distilgpt2`** - DistilGPT2 implementations
- **`distqat.models.gpt_neo`** - GPT-Neo implementations
- **`distqat.models.wav2vec2`** - Wav2Vec2 implementations

#### Optimization

- **`distqat.distributed.optim.diloco`** - DiLoCo optimizer implementation
- **`distqat.distributed.optim.collaborative`** - Collaborative optimization utilities

#### Utilities

- **`distqat.config`** - Configuration schemas and parsing
- **`distqat.data`** - Data loading and preprocessing
- **`distqat.sharding`** - Model sharding utilities

## 📝 Examples

### Basic Examples

1. **Local Training**
   ```bash
   python start_trainer_client.py
   ```

2. **Distributed Training (2 machines)**
   - Machine 1: `python start_trainer_client.py --public-ip $PUBLIC_IP`
   - Machine 2: `python start_servers.py --public-ip $PUBLIC_IP --num-servers 2 --initial-peers $PEERS`

3. **Failover Testing**
   ```bash
   python test_failover.py --public-ip $PUBLIC_IP
   ```

### Model Examples

- **ResNet18 on CIFAR-10** - Default configuration
- **ResNet50 on ImageNet** - Large-scale image classification
- **DistilGPT2** - Language model pretraining
- **GPT-Neo** - Larger language models
- **Wav2Vec2** - Audio/speech models

## 🧪 Testing

### Test Files

Located in `tests/` directory:

- `test_data_server.py` - Data server functionality
- Additional tests for models, utilities, etc.

### Running Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_data_server.py

# With coverage
pytest --cov=src/distqat tests/
```

## 🚢 Deployment

### Deployment Guides

- **[Deployment Overview](DEPLOYMENT.md)** - Supported targets and script generation workflows
- **[SkyPilot server setup template](skypilot/skypilot-server-setup.md)** - Cloud-init template for running the SkyPilot API server
- **Tested platforms**
  - [AWS AMD instances](skypilot/tested-platforms/aws-amd.md)
  - [AWS Lambda preview](skypilot/tested-platforms/lambda.md)

### SkyPilot Configurations

Located in `skypilot/` directory:

- `start_trainer_client.yaml` - Deploy coordinator node
- `start_servers.yaml` - Deploy worker nodes
- `demo.yaml` - Demo deployment
- `test_on_single_server.yaml` - Single-machine testing

### Launch Scripts

- `start_trainer_client.py` - Launch coordinator (monitor + client)
- `start_servers.py` - Launch worker servers
- `test_failover.py` - Test failover mechanism
- `run_demo.py` - Demo orchestration
- `run_local.py` - Swarm training script

## 🔍 Troubleshooting Resources

### Quick Fixes

From [README Troubleshooting section](README.md#troubleshooting-faq):

| Issue | Solution |
|-------|----------|
| Leftover processes | `pkill -f distqat` |
| Port conflicts | `kill -9 $(lsof -t -i:50000)` |
| CUDA OOM | Reduce `batch_size_per_step` |
| Slow training | Increase timeouts |
| WandB errors | `wandb login` |

### Detailed Guides

- **Network Issues** - DHT connectivity, firewall rules
- **Performance Issues** - Timeouts, bandwidth optimization
- **Checkpoint Issues** - Loading, saving, recovery
- **Configuration Issues** - YAML syntax, parameter tuning

## 📊 Monitoring

### Logging

Logs are saved to `logs/{experiment_prefix}/`:
- `monitor.log` - DHT and metrics
- `client.log` - Expert discovery
- `server_*.log` - Individual servers
- `trainer_*.log` - Individual trainers

### WandB Integration

Configure in YAML:
```yaml
wandb_project: "distqat" (if None, no WandB logging will happen)
wandb_entity: "your-entity" (can be None)
experiment_prefix: "your-experiment"
```

View metrics:
- Loss curves per trainer
- Samples per second
- Alive peers
- DHT state

## 🎓 Learning Resources

### Papers & References

- **Hivemind** - Decentralized deep learning
- **SWARM Parallelism** - Communication-efficient training
- **OpenDiLoCo** - Distributed low-communication optimization
- **Protocol Models** - Model parallelism over internet

See [Citations](README.md#citations) for full references.

### Related Projects

- **[Hivemind](https://github.com/learning-at-home/hivemind)** - P2P framework
- **[OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo)** - DiLoCo implementation
- **[Prime](https://github.com/PrimeIntellect-ai/prime)** - GPU resource management
- **[Petals](https://github.com/bigscience-workshop/petals)** - Distributed inference

## 📞 Support

### Getting Help

1. **Check Documentation** - Start with README and Quick Start
2. **Search Issues** - Look for similar problems on GitHub
3. **Ask Questions** - Open a discussion or issue
4. **Contact Maintainers** - See README for contact info

### Reporting Issues

When reporting issues, include:
- Python version and dependencies
- Configuration file
- Error messages and stack traces
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Describe the use case
- Explain expected behavior
- Provide examples if applicable

---

## 📄 License

All documentation is licensed under the same terms as the project (MIT License).

---

**Last Updated**: November 2025

For the most up-to-date documentation, see the [GitHub repository](https://github.com/yourorg/distqat).

