# Setup/Deployment

Since the deployments with this framework are inteded to be decentralized,
the deployment is not necessarily centrally managed.

The scripts in this directory provide a way to generate deployment scripts for
various target scenarios, and platforms.

## Supported operating systems

(these are the ones which should be tested, similar platforms will likely also work)

- **Ubuntu 22.04/24.04** _←← For now use Ubuntu 24.04, whenever available, other platforms are not well tested._
- (enterprise linux 8/9 (RHEL, CENT OS, Rocky Linux, Oracle Linux, etc.))
- (Amazon Linux)

## Supported setup/deployment mechanisms

- [cloud-init](https://cloud-init.io) scripts for deployment on any public/private cloud
- [skypilot](https://www.skypilot.co) launch configs for deploying to multi-cloud clusters
- single file bash script for simply and quickly deploying on any existing server
- archive of individual scripts for a given target platform

## SkyPilot resources

```{toctree}
:caption: SkyPilot Guides
:maxdepth: 1

skypilot/skypilot-server-setup
skypilot/tested-platforms/aws-amd
skypilot/tested-platforms/lambda
```

## Deployment

The primary command is `create-setup-files.py`, which creates setup scripts for a given config/machine.

**Please use `create-setup-files.py --help` to see up to date usage information**

At the moment, most deployments are only semi-automated (mainly providing a quick and easy way of generating the setup scrips, but the code to start and manage instances for a specific target cloud is up to the user — or just by using skypilot for supported platforms)

## Deployment Methods

The `create-setup-files.py` script supports 2 differnt deployment schemes:

- Single Stage: For lauching on known systems/already deployed (e.g. bare-metal) servers
- Two Stage: For launching on partly unknown systems (e.g. a heterogenous cluster of machines)

```mermaid
flowchart TB
    subgraph RIGHT["Two-stage deployment"]
    direction TB

    subgraph R1["Local machine"]
      R_setup_local[setup script]
    end

    subgraph R2["Target machine"]
      R_setup_local -->|"generates"| R_setup_1[first stage deploy script]
      R_setup_1 -->|"runs"| R_setup_target[setup script]
      R_setup_target -->|"invokes"| R_get2[get_initial_machine_info]
      R_get2 -. "machine_info" .-> R_setup_target
      R_setup_target -->|"generates"| R_stage2[second stage deploy script]
    end
    R_stage2 --> R_done([setup finished])
  end

  subgraph LEFT["Single-step deployment"]
    direction TB

    subgraph L1["Local machine"]
      L_setup[setup script]
    end

    subgraph L2["Target machine"]
      L_get[get_initial_machine_info]
      L_get -. "machine_info" .-> L_setup
      L_setup -->|"generates"| L_deploy[deploy script]
    end

      L_deploy --> L_done([setup finished])
  end

```

## Usage Examples

The directory `docs/tested-platforms` contains some specific tested manual deployment examples which can provide some context for platforms which you are familiar with.

### Auto detect configuration on existing machine

To make the configuration for the state of a specific target machine easier, you can run the script `get-initial-machine-info.sh` on any (non-root) shell on the target. (this is specifically designed to work well by just pasting in the entire file content to an interactive shell)

This will yield you something like:

```json
{
    "os": {
        "id": "rocky",
        "version_id": "8.10",
        "id_like": "rhel centos fedora",
        "platform_id": "platform:el8"
    },
    "cpu": {"architecture": "x86_64", "vendor": "AuthenticAMD", "model": "AMD EPYC 9355P 32-Core Processor", "cores": 32, "extensions": " fpu vme de ... "},
    "memory": {
        "total": 404723527680
    },
    "disk_info": {"root": {"total": 75125227520, "used": 75106054144, "free": 19173376}, "fhunstock": {"total": 2591831359488, "used": 956455452672, "free": 1635375906816}},
    "pci_devices": [{"address":"c1:00.0","class":"3D","vendor":"10de","device":"233b"}],
    "driver_info": {"nvidia": {"driver_version": "575.57.08", "cuda_version": "12.9"}, "amd": {"driver_version": null, "rocm_version": null}},
    "secure_boot_enabled": "false",
    "network": {"has_public_ip": false, "is_behind_nat": true, "external_ip": "79.254.12.142"},
    "user_info": {"username": "foobar", "has_sudo": "possible"}
}
```

Which you can use to create a deployment script specifically for that machine type.

e.g. for a one-of-a-kind deployment on e.g. a bare metal machine just sitting in your office, you can simply generate a script which you can just paste to an ssh session to set up the machine.

`setup/create-setup-files.py --machine-info <your machine info json> --local-private-key keys/ssk-key --clone-key keys/clone-key --target bash_script`

or to use the machine info to dump an initial config, to tweak (see next example right below)

`setup/create-setup-files.py --machine-info tmp/aws-amd-x4large.json --forward-authorized-keys --dump-config`

---

### Specify an existing config file, and generate a cloud-init file from it

Given the config file `base-config.json`

```json
{
  "os": {
    "id": "ubuntu",
    "version_id": "24.04",
    "platform_id": null
  },
  "create_new_user": false,
  "gpu_vendor": "nvidia",
  "install_drivers": false,
  "open_ports": [],
  "source_url": "git@github.com:SEMRON/aether.git",
  "username": "ubuntu",
  "management_user": null
}

```

you can use e.g.

`setup/create-setup-files.py --local-private-key keys/ssk-key --clone-key keys/clone-key --config base-config.json --target cloud_init`

to create a [cloud-config](https://cloud-init.io) which can be used with many cloud providers to spin up instances.
