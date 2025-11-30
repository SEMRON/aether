from typing import Iterable, List, Optional, Protocol, Dict, Any

from .common import *

def get_os_from_machine_info(machine_info: Dict[str, Any]) -> Optional[OS]:
    """Parse machine info JSON and return matching OS object."""
    if not isinstance(machine_info, dict) or "os" not in machine_info:
        return None

    os_info = machine_info["os"]
    if not isinstance(os_info, dict):
        return None

    # Get OS fields
    os_id = os_info.get("id")
    version_id = os_info.get("version_id")
    platform_id = os_info.get("platform_id")

    # Try to match Distro
    distro = None
    if os_id:
        # Direct match
        for d in Distro:
            if d.value == os_id.lower():
                distro = d
                break

    # Try to match Platform
    platform = None
    if platform_id:
        for p in Platform:
            if p.value == platform_id:
                platform = p
                break

    # Return OS object if we have at least some info
    if distro or version_id or platform:
        return OS(id=distro, version_id=version_id, platform_id=platform)

    return None

def get_gpu_vendor_from_machine_info(machine_info: Dict[str, Any]) -> Optional[GPU_VENDOR]:
    """Parse machine info JSON and return the GPU vendor if found."""
    if not isinstance(machine_info, dict) or "pci_devices" not in machine_info:
        return None

    pci_devices = machine_info["pci_devices"]
    if not pci_devices:
        return None

    # Collect all vendor IDs from PCI devices
    vendor_ids = set()
    for device in pci_devices:
        if isinstance(device, dict) and "vendor" in device:
            vendor_ids.add(device["vendor"])

    if not vendor_ids:
        return None

    # Check if all devices have the same vendor
    if len(vendor_ids) > 1:
        raise ValueError(f"Unclear vendor: found multiple PCI vendor IDs: {vendor_ids}")

    vendor_id = vendor_ids.pop()

    # Match vendor ID to GPU_VENDOR enum
    for gpu_vendor in GPU_VENDOR:
        if gpu_vendor.pci_vendor_id == vendor_id:
            return gpu_vendor

    return None

def validate_machine_info(machine_info_data: Dict[str, Any]) -> Exception|None:
    # Validate machine info structure
    required_keys = ["os", "pci_devices", "network", "user_info", "cpu", "driver_info", "memory", "disk_info", "secure_boot_enabled"]
    for key in required_keys:
        if key not in machine_info_data:
            return ValueError(f"Missing required key '{key}' in machine info")

    # Validate OS structure
    os_data = machine_info_data["os"]
    os_required_keys = ["id", "version_id", "id_like", "platform_id"]
    for key in os_required_keys:
        if key not in os_data:
            return ValueError(f"Missing required key 'os.{key}' in machine info")

    # Validate cpu structure
    cpu_data = machine_info_data["cpu"]
    cpu_required_keys = ["architecture", "model", "cores", "extensions", "vendor"]
    for key in cpu_required_keys:
        if key not in cpu_data:
            return ValueError(f"Missing required key 'cpu.{key}' in machine info")

    # Validate pci_devices structure
    pci_devices = machine_info_data["pci_devices"]
    if not isinstance(pci_devices, list):
        return ValueError("pci_devices must be a list")
    for i, device in enumerate(pci_devices):
        if not isinstance(device, dict):
            return ValueError(f"pci_devices[{i}] must be a dictionary")
        pci_required_keys = ["address", "class", "vendor", "device"]
        for key in pci_required_keys:
            if key not in device:
                return ValueError(f"Missing required key 'pci_devices[{i}].{key}' in machine info")

    # Validate driver_info structure
    driver_info_data = machine_info_data["driver_info"]
    driver_info_required_keys = ["nvidia", "amd"]
    for key in driver_info_required_keys:
        if key not in driver_info_data:
            return ValueError(f"Missing required key 'driver_info.{key}' in machine info")

    # Validate nvidia driver info
    nvidia_data = driver_info_data["nvidia"]
    if not isinstance(nvidia_data, dict):
        return ValueError("driver_info.nvidia must be a dictionary")
    nvidia_required_keys = ["driver_version", "cuda_version"]
    for key in nvidia_required_keys:
        if key not in nvidia_data:
            return ValueError(f"Missing required key 'driver_info.nvidia.{key}' in machine info")

    # Validate amd driver info
    amd_data = driver_info_data["amd"]
    if not isinstance(amd_data, dict):
        return ValueError("driver_info.amd must be a dictionary")
    amd_required_keys = ["driver_version", "rocm_version"]
    for key in amd_required_keys:
        if key not in amd_data:
            return ValueError(f"Missing required key 'driver_info.amd.{key}' in machine info")

    # Validate secure_boot_enabled
    secure_boot = machine_info_data["secure_boot_enabled"]
    if secure_boot not in ["true", "false", "unknown"]:
        return ValueError(f"secure_boot_enabled must be 'true', 'false', or 'unknown', got: '{secure_boot}'")

    # Validate memory structure
    memory_data = machine_info_data["memory"]
    if not isinstance(memory_data, dict):
        return ValueError("memory must be a dictionary")
    if "total" not in memory_data:
        return ValueError("Missing required key 'memory.total' in machine info")
    if not isinstance(memory_data["total"], (int, float)):
        return ValueError("memory.total must be a number")

    # Validate disk_info structure
    disk_info_data = machine_info_data["disk_info"]
    if not isinstance(disk_info_data, dict):
        return ValueError("disk_info must be a dictionary")
    if "root" not in disk_info_data:
        return ValueError("Missing required key 'disk_info.root' in machine info")

    root_data = disk_info_data["root"]
    if not isinstance(root_data, dict):
        return ValueError("disk_info.root must be a dictionary")
    root_required_keys = ["total", "used", "free"]
    for key in root_required_keys:
        if key not in root_data:
            return ValueError(f"Missing required key 'disk_info.root.{key}' in machine info")
        if not isinstance(root_data[key], (int, float)):
            return ValueError(f"disk_info.root.{key} must be a number")

    # Validate network structure
    network_data = machine_info_data["network"]
    network_required_keys = ["has_public_ip", "is_behind_nat", "external_ip"]
    for key in network_required_keys:
        if key not in network_data:
            return ValueError(f"Missing required key 'network.{key}' in machine info")

    # Validate user_info structure
    user_info_data = machine_info_data["user_info"]
    user_info_required_keys = ["username", "has_sudo"]
    for key in user_info_required_keys:
        if key not in user_info_data:
            return ValueError(f"Missing required key 'user_info.{key}' in machine info")

    return None
