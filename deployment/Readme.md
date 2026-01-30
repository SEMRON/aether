## Sub Apps

For testing purposes, there are a number of sub-components which can be run
as their own independant programs from the command line.

*you of course have to load the `venv` for any of these to work*

### Deployment

`deploy.py` is used to provision servers according to a provided
setup definition.

```bash
# Run from the deployment/ directory

# to deploy/provision servers
python3 deploy.py deploy --setup-definition <setup definition file>

# to destroy/unprovision servers
python3 deploy.py destroy --setup-definition <setup definition file> <filter>

# the filter specifies which instances you want to destroy
# see `--help` for the available filters
```

Filters for `destroy`

| Filter | Description |
|-|-|
| `all` | Destroy all systems |
| `aws-inst` | Destroy a specific node (as named in the inventory file) |
| `g=smallvm` | Destroy all systems which are part of the "smallvm" group. A group might be a runner class, os type, or deployment provider. See the inventory file for the available groups |
| `depl=<..>` | During each deployment, every node created is added to a randomly (GUID) named deployment group for that command invocation. You can use this to destroy all nodes you mistakenly created by running the last deploy |

See `examples/all-variants.toml` for an example setup definition.

The *setup definition file* is used to specify a target deployment, and the
*inventory file* is used to describe the *actual servers* currently deployed.

#### Supported Providers

- **libvirt** - Local KVM/QEMU virtualization for testing
- **prime_intellect** - Prime Intellect GPU cloud API

See `examples/prime-intellect.toml` for a Prime Intellect specific example.

#### Inventory tracking

The deployment keeps track of the deployed machines in an `inventory.yaml`
file, which follows the [Ansible](https://docs.ansible.com/ansible/latest/inventory_guide/intro_inventory.html)
style.

**This same file is used by e.g. `run-tasks.py` to find the nodes
and get node-specific configuration variables.**

By default, this inventory file is stored in
`workdir_deploy/inventory.yaml`.

**Be careful not to corrupt this file, as you may need it to keep track of what
servers you hve provisioned for the current project**

#### Authentication

By default, the deploy command sets up all ssh puplic keys
**and all authorized keys** from the current users `~/.ssh` directory
as authorized keys for the main user on the deployed servers.

For Prime Intellect deployments, set your API token:
```bash
export PRIME_INTELLECT_API_TOKEN=<your-token>
```

### Source Deployment Server

The `host-server` serves the `.git` directory of this repo from the
current host, for devices to pull the current sources.

```bash
python3 host-server.py
```

This means, to deploy an application to the runners, you always need to commit
the source which you want to run. This ensures that every runner, and the host
use the same application state, and don't get out of sync.

The server simply hosts the local repos `.git` directory over https.
For authentication, an auto-generated pre-shared-key is used (for runner auth),
and ther server uses an also auto-generated self-signed certificate, which the
runners can use to authenticate the master node.

By default, the server cert, private key, and psk are stored in
`workdir_server`.
You can specify a custom working directory for the server, to use a private key
and certificate you provide for deployment server authentication,
make sure to also provide this certificate to the clients  during deployment.
(set `aether_deployment_server_cert` for deployment)

### Deployment Tasks

`run-tasks.py` is a helper app, which executes ansible tasks,
while providing the appropriate variables, and using the inventory generated
by the deployment scripts.

```bash
python3 run-tasks.py [<task name/path>, ..]

# for more custom use, see `--help`

# list the available tasks
ls ansible-tasks
# Note: for these you can just specify the base of the filename, e.g.
#       wait-init is equivalent to ansible-tasks/wait-init.yaml
```

Tasks are executed by Ansible in the order in which they are specified on the
command line.

See `templates/wrapper-playbook.yaml` for the variables you can set
for more custom use.

#### Subtasks

| Name | Description |
|-|-|
| `wait-init` | Wait until the initial cloud-init from deployment is finished |
| `packages` | Install the appropriate dependencies (for the OS of the host) |
| `deploy-service` | Deploy a local executable as a systemd service |
| `service-status` | Check the status of a deployed service |
| `stop-service` | Stop (and optionally remove) a deployed service |

#### Service Management

The `deploy-service`, `service-status`, and `stop-service` tasks allow you to
deploy and manage executables as systemd services on remote hosts.

**Deploy a service:**

```bash
# Basic deployment
python3 run-tasks.py deploy-service \
    -D SERVICE_NAME=myapp \
    -D LOCAL_FILE_PATH=/path/to/myapp

# With additional options
python3 run-tasks.py deploy-service \
    -D SERVICE_NAME=myapp \
    -D LOCAL_FILE_PATH=/path/to/myapp \
    -D SERVICE_USER=nobody \
    -D SERVICE_ARGS="--port 8080" \
    -D SERVICE_WORKING_DIR=/var/lib/myapp
```

Variables for `deploy-service`:
| Variable | Required | Default | Description |
|-|-|-|-|
| `SERVICE_NAME` | Yes | - | Name of the systemd service |
| `LOCAL_FILE_PATH` | Yes | - | Path to executable on local machine |
| `REMOTE_INSTALL_DIR` | No | `/opt/{SERVICE_NAME}` | Installation directory |
| `SERVICE_USER` | No | `root` | User to run the service as (use `ansible_user` for the connection user) |
| `SERVICE_ARGS` | No | `""` | Command line arguments |
| `SERVICE_WORKING_DIR` | No | `REMOTE_INSTALL_DIR` | Working directory |
| `SERVICE_RESTART` | No | `on-failure` | Systemd restart policy |
| `SERVICE_ENV` | No | `[]` | Environment variables (list of `key=value`) |

**Check service status:**

```bash
python3 run-tasks.py service-status -D SERVICE_NAME=myapp
```

Shows active state, PID, memory usage, and recent logs.

**Stop a service:**

```bash
# Just stop
python3 run-tasks.py stop-service -D SERVICE_NAME=myapp

# Stop and disable
python3 run-tasks.py stop-service -D SERVICE_NAME=myapp -D DISABLE_SERVICE=yes

# Stop, disable, and remove completely
python3 run-tasks.py stop-service -D SERVICE_NAME=myapp -D REMOVE_SERVICE=yes
```

#### Example: Heartbeat Service

Deploy a simple heartbeat script that sends periodic notifications to ntfy.sh:

```bash
# Deploy the heartbeat service to all nodes
python3 run-tasks.py deploy-service \
    -D SERVICE_NAME=heartbeat \
    -D LOCAL_FILE_PATH=../examples/heartbeat-ntfy.sh

# Check if it's running
python3 run-tasks.py service-status -D SERVICE_NAME=heartbeat

# View heartbeats at: https://ntfy.sh/oYhIC0qan6FDTxsk

# Stop and remove the service when done
python3 run-tasks.py stop-service -D SERVICE_NAME=heartbeat -D REMOVE_SERVICE=yes
```

Note: File paths in `LOCAL_FILE_PATH` are relative to the `ansible-tasks/` directory,
so use `../examples/` to reference files in the examples folder.

## Requirements

```bash
# optional, for development/testing
apt-get -y install genisoimage cockpit cockpit-system cockpit-machines
```

### Libvirt

Libvirt is used to test deployment procedures, and network configs,
without having to spin-up cloud instances. (with the associated costs)

It uses the current host as a virtualization server, using KVM and qemu.

If you are running on a cloud server, make sure your cloud provider allows
(nested) virtualization on the instance you are running on.

Setup:
```bash
# enable ipv4 forwarding
sudo sed -i 's/^#net.ipv4.ip_forward/net.ipv4.ip_forward/' /etc/sysctl.conf

# configure the bridge connection
sudo mkdir -p /etc/qemu
sudo bash -c 'echo "allow virbr0" > /etc/qemu/bridge.conf'
sudo chown root:kvm /etc/qemu/bridge.conf
sudo chmod 0660 /etc/qemu/bridge.conf

# create storage location for vm images
sudo mkdir /opt/aether-vm
sudo chown $USER:kvm /opt/aether-vm
```

Dependencies:

```bash
apt-get -y install qemu-kvm bridge-utils
```
