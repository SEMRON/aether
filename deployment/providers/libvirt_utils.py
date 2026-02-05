import datetime
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

from utils import cloud_init, imagegen_utils, ssh_key_utils


def _use_system_package(package_name):
    """Add the system path for a specific package to sys.path"""
    # Run a subprocess to find where the package is in the system Python
    cmd = [
        f"/usr/bin/{os.path.basename(sys.executable)}",
        "-c",
        f"import {package_name}; print({package_name}.__file__)",
    ]
    try:
        # Run outside the virtual environment
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        package_path = result.stdout.strip()
        # Extract the directory and add to path
        package_dir = os.path.dirname(package_path)
        if package_dir not in sys.path:
            sys.path.append(package_dir)
        return True
    except subprocess.CalledProcessError:
        return False


# Only load libvirt if ENABLE_LIBVIRT is set
libvirt = None
if os.environ.get("ENABLE_LIBVIRT"):
    if not _use_system_package("libvirt"):
        raise RuntimeError("Unable to load libvirt python library from system.")
    import libvirt  # pyright: ignore
else:
    print("Note: libvirt support disabled. Set ENABLE_LIBVIRT=1 to enable.")


def _check_libvirt_available():
    """Check if libvirt is available, raise error if not."""
    if libvirt is None:
        raise RuntimeError(
            "libvirt is not available. Set ENABLE_LIBVIRT=1 environment variable to enable."
        )


def get_connection():
    _check_libvirt_available()
    conn = libvirt.open("qemu:///system")
    if conn is None:
        raise RuntimeError("Failed to open connection to qemu:///system")
    return conn


def close_connection(conn):
    conn.close()


@dataclass
class DHCPLease:
    clientid: str
    expirytime: int
    hostname: Optional[str]
    iaid: Optional[str]
    iface: str
    ipaddr: str
    mac: str
    prefix: int
    type: int

    @property
    def expires_at(self):
        return datetime.datetime.fromtimestamp(self.expirytime)


def get_dhcp_leases(conn):
    network = conn.networkLookupByName("default")
    if network is None:
        print("Failed to find the 'default' network")
        sys.exit(1)

    # Get the DHCP leases
    leases = network.DHCPLeases()

    lease_list = [DHCPLease(**lease) for lease in leases]

    return lease_list


class DomainState(Enum):
    RUNNING = "running"
    SHUT_OFF = "shut off"


@dataclass
class DomainInfo:
    id: int
    name: str
    uuid: str
    state: DomainState


def list_running_domains(conn):
    """Lists running domains only."""
    domain_ids = conn.listDomainsID()
    result = []
    for dom_id in domain_ids:
        dom = conn.lookupByID(dom_id)
        state, _, _, _, _ = dom.info()
        result.append(
            DomainInfo(
                id=dom_id,
                name=dom.name(),
                uuid=dom.UUIDString(),
                state=DomainState.RUNNING,
            )
        )
    return result


def get_mac_address(conn, domain_name: str):
    """Gets MAC address from a domain's configuration."""
    domain = conn.lookupByName(domain_name)
    xml = domain.XMLDesc()
    tree = ET.fromstring(xml)

    network_interfaces = tree.findall(".//interface[@type='network']")
    if len(network_interfaces) != 1:
        raise RuntimeError(
            f"Expected exactly one network interface for domain {domain_name}, found {len(network_interfaces)}"
        )

    mac_element = network_interfaces[0].find("mac")
    if mac_element is None:
        raise RuntimeError(
            f"No MAC address found for network interface in domain {domain_name}"
        )

    mac = mac_element.get("address")

    if mac is None:
        raise RuntimeError(f"Formatting error for domain XML {domain_name}")

    return mac


def list_inactive_domains(conn):
    """Lists inactive (shut off) domains only."""
    defined_names = conn.listDefinedDomains()
    result = []
    for name in defined_names:
        dom = conn.lookupByName(name)
        result.append(
            DomainInfo(
                id=-1,
                name=name,
                uuid=dom.UUIDString(),
                state=DomainState.SHUT_OFF,
            )
        )
    return result


def list_domains(conn):
    """Lists all domains (running and shut off), similar to `virsh list --all`."""
    result = []
    result.extend(list_running_domains(conn))
    result.extend(list_inactive_domains(conn))
    return result


def destroy_domain(conn, name):
    """Forcibly shuts down (destroys) a running domain."""
    domain = conn.lookupByName(name)
    if domain is None:
        raise ValueError(f"Domain '{name}' not found.")

    domain.destroy()


def get_disks(conn, domain_name):
    domain = conn.lookupByName(domain_name)
    xml = domain.XMLDesc()
    tree = ET.fromstring(xml)
    disks = tree.findall(".//disk")
    return disks


def delete_storage_pool_by_path(conn, pool_path: str):
    """Deletes a storage pool by its path.

    Args:
        conn: Libvirt connection
        pool_path: Path to the storage pool to delete

    Raises:
        ValueError: If no pool found at the given path
        RuntimeError: If pool deletion fails
    """
    # Find pool matching the path
    pool = None
    try:
        pool = conn.storagePoolLookupByTargetPath(pool_path)
    except libvirt.libvirtError:
        pass

    if pool is None:
        raise ValueError(f"No storage pool found at path: {pool_path}")

    try:
        if pool.isActive():
            pool.destroy()
        pool.undefine()
    except libvirt.libvirtError as e:
        raise RuntimeError(f"Failed to delete storage pool: {e}")

    print(f"Storage pool at {pool_path} has been deleted")


def undefine_domain(conn, name: str, remove_all_storage: bool):
    """Undefines a domain. Optionally removes all associated storage."""
    domain = conn.lookupByName(name)
    if domain is None:
        raise ValueError(f"Domain '{name}' not found.")

    if remove_all_storage:
        disks = get_disks(conn, name)

        domain_dir = None

        for disk in disks:
            source = disk.find("source")
            if source is not None:
                disk_path = Path(str(source.get("file")))
                if name == disk_path.parent.name:
                    if domain_dir and domain_dir != disk_path.parent:
                        raise RuntimeError(
                            "Assumed the disk images to be deleted are all "
                            + "stored in the same path but found multiple "
                            + "paths with the name of the domain:\n"
                            + f"{domain_dir} and {disk_path.parent}"
                        )
                    domain_dir = disk_path.parent
                    try:
                        os.remove(disk_path)
                        print(f"Removed storage file: {disk_path}")
                    except OSError as e:
                        print(f"Failed to remove storage file {disk_path}: {e}")
                else:
                    print(
                        "Skipping storage file not in domain directory:"
                        + str(disk_path)
                    )

        # Check if domain directory is empty and remove if possible
        if domain_dir and domain_dir.exists():
            try:
                remaining_files = list(domain_dir.iterdir())
                if not remaining_files:
                    delete_storage_pool_by_path(conn, str(domain_dir))
                    domain_dir.rmdir()
                    print(f"Removed empty domain directory: {domain_dir}")
                else:
                    print(
                        f"Warning: Domain directory {domain_dir} still "
                        + f"contains files: {[f.name for f in remaining_files]}"
                    )
            except OSError as e:
                print(f"Failed to remove domain directory {domain_dir}: {e}")

        domain.undefineFlags(
            libvirt.VIR_DOMAIN_UNDEFINE_MANAGED_SAVE
            | libvirt.VIR_DOMAIN_UNDEFINE_SNAPSHOTS_METADATA
            | libvirt.VIR_DOMAIN_UNDEFINE_NVRAM
        )
    else:
        domain.undefine()


def wait_for_vm_ips(
    conn,
    expected_vms: List[str],
    poll_interval: int = 5,
    timeout: int = 300,
    verbose: bool = False,
) -> dict:
    """Check VM status and get IPs for a list of expected VMs.

    Args:
        conn: Libvirt connection
        expected_vms: List of VM names/hostnames expected to be running
        poll_interval: How often to check for IPs in seconds
        timeout: Maximum time to wait for IPs in seconds
        verbose: Whether to print progress updates

    Returns:
        Dict mapping VM names to their IP addresses

    Raises:
        RuntimeError: If any VMs are not defined or not running
    """
    domains = list_domains(conn)

    if verbose:
        print(f"running domains: {domains}")

    domain_map = {d.name: d for d in domains}

    # Transform names for hostname compatibility
    mac_to_expected_vms: Dict[str, str] = {
        get_mac_address(conn, dn): dn for dn in expected_vms
    }

    # Check if all VMs are defined
    undefined = []
    for vm in expected_vms:
        if vm not in domain_map:
            undefined.append(vm)
    if undefined:
        raise RuntimeError(f"VMs not defined: {', '.join(undefined)}")

    # Check if all VMs are running
    not_running = []
    for vm in expected_vms:
        if domain_map[vm].state != DomainState.RUNNING:
            not_running.append(vm)
    if not_running:
        raise RuntimeError(f"VMs not running: {', '.join(not_running)}")

    # Poll for IPs until we have them all or timeout
    start_time = datetime.datetime.now()
    result = {}

    while len(result) < len(mac_to_expected_vms):
        current_time = datetime.datetime.now()
        if (current_time - start_time).total_seconds() > timeout:
            missing = set(mac_to_expected_vms.keys()) - set(result.keys())
            raise RuntimeError(
                f"Timeout waiting for IPs. Missing: {', '.join(missing)}"
            )

        leases = get_dhcp_leases(conn)
        for lease in leases:
            if (
                lease.mac in mac_to_expected_vms
                and lease.hostname not in result
            ):
                inventory_name = mac_to_expected_vms[lease.mac]
                result[inventory_name] = lease.ipaddr
                if verbose:
                    print(
                        f"[{len(result):3d}/"
                        + f"{len(mac_to_expected_vms):3d}] "
                        + f"detected >{inventory_name}< as "
                        + f">{lease.hostname}< >{lease.mac}< "
                        + f"at >{lease.ipaddr}<"
                    )

        if len(result) < len(mac_to_expected_vms):
            if poll_interval > 0:
                time.sleep(poll_interval)

    return result


def define_and_start_vm(
    conn,
    name: str,
    ram: str,
    vcpus: int,
    disk_path: str,
    cdrom_path: str,
    os_variant: Optional[str],
    extraopts: List[str] = [],
):
    xml = generate_vm_xml(
        name=name,
        ram=ram,
        vcpus=vcpus,
        disk_path=disk_path,
        cdrom_path=cdrom_path,
        os_variant=os_variant,
        extraopts=extraopts,
    )
    try:
        domain = conn.defineXML(xml)
        if domain is None:
            raise RuntimeError(f"Failed to define domain '{name}'")
        domain.create()
        print(f"VM '{name}' has been defined and started.")
    except libvirt.libvirtError as e:
        raise RuntimeError(f"Libvirt error: {e}")


def generate_vm_xml(
    name: str,
    ram: str,
    vcpus: int,
    disk_path: str,
    cdrom_path: str,
    os_variant: Optional[str] = None,
    extraopts: List[str] = [],
) -> str:
    cmd = [
        "virt-install",
        "--connect",
        "qemu:///system",
        "--print-xml",
        "--name",
        name,
        "--ram",
        ram,
        "--vcpus",
        str(vcpus),
        "--import",
        "--disk",
        f"path={disk_path},format=qcow2",
        "--disk",
        f"path={cdrom_path},device=cdrom",
        "--network",
        "network=default",
        "--noautoconsole",
    ]

    if os_variant:
        cmd.extend(["--os-variant", os_variant])

    cmd.extend(extraopts)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error generating XML: {e.stderr}")


def create_vm_disk_image(
    output_path: str, base_image: str, size: str, format: str = "qcow2"
):
    """
    Creates a QCOW2 image with a backing file using qemu-img.
    """
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Target image '{output_path}' already exists.")

    cmd = [
        "qemu-img",
        "create",
        "-b",
        base_image,
        "-f",
        format,
        "-F",
        format,
        output_path,
        size,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Created image: {output_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create image: {e}")


def create_vm(
    conn,
    vm_storage_path: str,
    base_image_name: str,
    instance_name: str,
    ram: str,
    vcpus: int,
    disk_size: str,
    username: str,
    os_variant: Optional[str] = None,
    ssh_keys: List[str] | None = None,
    root_password: Optional[str] = None,
):
    """
    Creates a new VM with the given parameters.

    Args:
        conn: Libvirt connection
        vm_storage_path: Base path where VM storage is kept
        base_image_name: Name of the base image file (should be in vm_storage_path)
        instance_name: Name for the new VM instance
        ram: RAM size (format: '2048' for MB or '2G' for GB)
        vcpus: Number of virtual CPUs
        disk_size: Size of the disk (format: '20G')
        ssh_keys: List of SSH public keys to add to the instance
        username: Username for the main user account
        root_password: Optional root password
        os_variant: OS variant for the VM
    """
    # Create a directory for this VM's files
    vm_dir = Path(vm_storage_path) / instance_name
    vm_dir.mkdir(parents=True, exist_ok=False)

    # Paths for VM files
    base_image = str(Path(vm_storage_path) / base_image_name)
    disk_path = str(vm_dir / f"{instance_name}-disk.qcow2")
    cloud_init_iso = str(vm_dir / f"{instance_name}-cloud-init.iso")

    # Create the disk image
    create_vm_disk_image(
        output_path=disk_path, base_image=base_image, size=disk_size
    )

    # Generate cloud-init ISO
    imagegen_utils.create_cloud_init_iso(
        iso_path=cloud_init_iso,
        user_data_fn=lambda: cloud_init.render_cloud_init_user_data(
            username=username,
            ssh_keys=ssh_key_utils.these_or_all_keys(ssh_keys),
            root_password=root_password,
        ),
        meta_data_fn=lambda: cloud_init.render_cloud_init_meta_data(
            instance_name=instance_name
        ),
    )

    # Define and start the VM
    define_and_start_vm(
        conn=conn,
        name=instance_name,
        ram=ram,
        vcpus=vcpus,
        disk_path=disk_path,
        cdrom_path=cloud_init_iso,
        os_variant=os_variant,
    )

    print(f"VM '{instance_name}' has been created and started.")
