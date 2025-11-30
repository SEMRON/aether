| Cloud            | [Lambda Labs](https://cloud.lambda.ai) |
| ---------------- | -------------------------------------- |
| Instance Type    | gpu_1x_a10                             |
| CPU Architecture | x86_64                                 |
| Image            | Ubuntu 24.04                           |
| Accelerator      | Nvidia A10                             |

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
    "vendor": "GenuineIntel",
    "model": "Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz",
    "cores": 30,
    "extensions": " fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd arat vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm md_clear arch_capabilities"
  },
  "memory": {
    "total": 238546042880
  },
  "disk_info": {
    "root": {
      "total": 1455042297856,
      "used": 1717317632,
      "free": 1453308203008
    }
  },
  "pci_devices": [
    { "address": "06:00.0", "class": "3D", "vendor": "10de", "device": "2236" }
  ],
  "driver_info": {
    "nvidia": { "driver_version": null, "cuda_version": null },
    "amd": { "driver_version": null, "rocm_version": null }
  },
  "secure_boot_enabled": "false",
  "network": {
    "has_public_ip": false,
    "is_behind_nat": true,
    "external_ip": "170.9.12.110"
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
    "ssh-ed25519 AAAAC... aether@notebook"
  ],
  "create_new_user": false,
  "deploy_key_private": "-----BEGIN OPENSSH PRIVATE KEY-----\nb.....\n-----END OPENSSH PRIVATE KEY-----\n",
  "deploy_key_public": "ssh-rsa AAAAB.... clone key\n",
  "extras": {},
  "gpu_vendor": "nvidia",
  "install_drivers": true,
  "open_ports": [],
  "source_url": "git@github.com:SEMRON/distqat.git",
  "username": "ubuntu",
  "management_user": null
}
```
