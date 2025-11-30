| Cloud            | AWS                                                                     |
| ---------------- | ----------------------------------------------------------------------- |
| Instance Type    | g4ad.xlarge                                                             |
| CPU Architecture | x86_64                                                                  |
| Image (AMI)      | ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-20251022 Gen2 |
| Accelerator      | Radeon Pro V520                                                         |
| Storage          | 64 GiB _— Non Default !!!!_                                             |

## Machine Info:

```json
{
  "os": {
    "id": "ubuntu",
    "version_id": "24.04",
    "id_like": "debian",
    "platform_id": null
  },
  "cpu": {
    "architecture": "x86_64",
    "vendor": "AuthenticAMD",
    "model": "AMD EPYC 7R32",
    "cores": 4,
    "extensions": " fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch topoext ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr rdpru wbnoinvd arat npt nrip_save rdpid"
  },
  "memory": {
    "total": 16110223360
  },
  "disk_info": {
    "root": { "total": 65445814272, "used": 1890160640, "free": 63538876416 }
  },
  "pci_devices": [
    {
      "address": "00:1e.0",
      "class": "Disp",
      "vendor": "1002",
      "device": "7362"
    }
  ],
  "driver_info": {
    "nvidia": { "driver_version": null, "cuda_version": null },
    "amd": { "driver_version": null, "rocm_version": null }
  },
  "secure_boot_enabled": "false",
  "network": {
    "has_public_ip": false,
    "is_behind_nat": true,
    "external_ip": "18.130.50.114"
  },
  "user_info": { "username": "ubuntu", "has_sudo": "true" }
}
```

## Example config

```json
{
  "os": {
    "id": "ubuntu",
    "version_id": "24.04",
    "platform_id": null
  },
  "authorized_pubkey": [
    "ssh-ed25519 AAAAC.... aether@host"
  ],
  "create_new_user": false,
  "deploy_key_private": "-----BEGIN OPENSSH PRIVATE KEY-----\n.....\n-----END OPENSSH PRIVATE KEY-----\n",
  "deploy_key_public": "ssh-rsa AAAAB.... clone key\n",
  "extras": {},
  "gpu_vendor": "amd",
  "install_drivers": true,
  "open_ports": [],
  "source_url": "git@github.com:SEMRON/aether.git",
  "username": "ubuntu",
  "management_user": "aetheradm"
}
```

# Manual setup example

For a sanity check, these are the instructions to replicate a setup

- open the [AWS cloud console → Instances](https://console.aws.amazon.com/ec2/home#Instances:) page for a region to which you have access
- click on (Launch Instances)
- enter a name, the name is not relevant
- select the image type ("Amazon Machine Image (AMI)") as _Ubuntu Server 24.04 LTS_
- select the Instance type as "g4ad.xlarge"
- select a key pair (this is required by AWS, but the generated cloud-init script actually adds your keys also)
- under "Configure storage" select at least 50 GiB of storage (the ROCm installation uses a lot of storage)
- open "Advanced details"
  - if you only have an AWS quota for Spot instances, select as such under "Purchasing option"
  - **(for cloud-init)** under "User data" paste in the cloud-init file, \
 e.g generated with:\
 `setup/create-setup-files.py --machine-info <machine info> -k keys/ssh-key -ck keys/clone-key -t cloud_init --management-user aetheradm`\
 → Launch the instance
- **(for interactive/debugging setup)**
  - launch the instance as-is
  - connect using the key previously selected under "Key pair (login)" to `ubuntu@<public ip>`
  - use the `get-initial-machine-info.sh` script by simply pasting the entire script into the command line, to gather system info (copy the result json block)
  - run the setup scrip generator: (paste in the machine info when asked):\
  `./create-setup-files.py -k keys/ssk-key -ck keys/clone-key -t bash_script`
  - take the output script, and simply paste it into the interactive ssh session to start the setup
