import argparse
import copy
import os
import tomllib
import uuid
from pprint import pprint
from typing import Any, Dict, Optional

from providers import prime_intellect
from utils import config, inventory

# Conditionally import libvirt_utils
libvirt_utils = None
if os.environ.get("ENABLE_LIBVIRT"):
    from providers import libvirt_utils


def read_toml_config_file(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def destroy(inventory_file_path: str, filters: list[str], setup_definition_file_path: Optional[str] = None):
    """
    A node will be destroyed when it matches a node name filter or is part of
    _all_ groups which were specified.
    "all" deletes all nodes (scoped to setup definition if provided).
    Filter types (and their syntax):
        node_name: <literal_node_name>
        group_name: g=<literal_group_name>
        guid: depl=<literal_deployment_guid>
        status: s=<status> (e.g., s=ACTIVE, s=FAILED, s=TERMINATED, s=UNKNOWN)
        all: all (scoped to nodes in setup definition if provided)
    """
    assert filters

    inv = inventory.NodeInventory.from_file(inventory_file_path)

    # Load setup definition to scope filters (if provided)
    config_node_names: Optional[set[str]] = None
    if setup_definition_file_path:
        parsed_nodes = list(
            config.parse_nodes(read_toml_config_file(setup_definition_file_path))
        )
        config_node_names = {node.name for node in parsed_nodes}
        print(f"Setup definition '{setup_definition_file_path}' defines {len(config_node_names)} node(s): {sorted(config_node_names)}")
    else:
        print("No setup definition provided - 'all' will destroy all nodes in inventory")

    # Extract and check group filters
    group_filters = [f[2:] for f in filters if f.startswith("g=")]
    depl_filters = [f[5:] for f in filters if f.startswith("depl=")]
    status_filters = [f[2:].upper() for f in filters if f.startswith("s=")]

    # If status filters are present, fetch statuses for Prime Intellect nodes
    pod_statuses: Dict[str, Optional[prime_intellect.PodStatus]] = {}
    if status_filters:
        pi_nodes = [n for n in inv.nodes if n.vars.get("provider") == "prime_intellect"]
        pod_ids = []
        for n in pi_nodes:
            pod_id = n.vars.get("pod_id")
            if pod_id is not None:
                pod_ids.append(str(pod_id))
        if pod_ids:
            pi_client = prime_intellect.PrimeIntellectClient()
            pod_statuses = pi_client.get_pod_statuses(pod_ids)
            pprint(pod_statuses)
    nodes_to_destroy = list()
    for node in inv.nodes:
        # Process "all" filter first - scoped to nodes in setup definition if provided
        if "all" in filters:
            if config_node_names is None or node.name in config_node_names:
                nodes_to_destroy.append(node)
            continue

        # Process node name filter - validate against setup definition if provided
        if node.name in filters:
            if config_node_names is not None and node.name not in config_node_names:
                print(f"Warning: Node '{node.name}' not in setup definition, skipping")
                continue
            nodes_to_destroy.append(node)
            continue

        # Node must match ALL specified groups
        matches_all_groups = len(group_filters) > 0
        if group_filters:
            for group_name in group_filters:
                if group_name not in node.groups:
                    matches_all_groups = False
                    break

        # Node must match at least one deployment GUID if any are specified
        matches_depl = not depl_filters  # True if no depl filters specified
        if depl_filters:
            for deployment_guid in depl_filters:
                if node.guid == deployment_guid:
                    matches_depl = True
                    break

        # Check status filter for Prime Intellect nodes
        matches_status = not status_filters  # True if no status filters specified
        if status_filters:
            provider = node.vars.get("provider")
            if provider == "prime_intellect":
                pod_id = node.vars.get("pod_id")
                pod_status = pod_statuses.get(pod_id) if pod_id else None
                if pod_status:
                    if pod_status.status.upper() in status_filters:
                        matches_status = True
                elif "UNKNOWN" in status_filters:
                    # No status available = UNKNOWN
                    matches_status = True
            else:
                # Non-Prime Intellect nodes: only match if UNKNOWN is in filter
                if "UNKNOWN" in status_filters:
                    matches_status = True

        # Add node if it matches all criteria
        if matches_all_groups and matches_depl and matches_status:
            nodes_to_destroy.append(node)

    if len(nodes_to_destroy) == 0:
        raise RuntimeWarning("No nodes to destroy. Consider that non-group/name filters, need a group/list of names to work. Consider adding 'all'.")

    # Separate nodes by provider type based on their vars
    libvirt_nodes = []
    prime_intellect_nodes = []

    for node in nodes_to_destroy:
        provider = node.vars.get("provider")
        if provider == "prime_intellect":
            prime_intellect_nodes.append(node)
        else:
            # Default to libvirt for backward compatibility
            libvirt_nodes.append(node)

    # Destroy libvirt nodes
    if libvirt_nodes:
        if libvirt_utils is None:
            print("Error: libvirt support not enabled. Set ENABLE_LIBVIRT=1 to enable.")
            print(f"Cannot destroy {len(libvirt_nodes)} libvirt node(s)")
        else:
            conn = libvirt_utils.get_connection()
            for node in libvirt_nodes:
                try:
                    libvirt_utils.destroy_domain(conn, node.name)
                    print(f"shut down node '{node.name}'")
                except libvirt_utils.libvirt.libvirtError:
                    print(f"failed to shut down node '{node.name}'")

                try:
                    libvirt_utils.undefine_domain(
                        conn, node.name, remove_all_storage=True
                    )
                    inv.remove_nodes(lambda inv_node: inv_node.name == node.name)
                    print(f"Deleted node '{node.name}'")
                    inv.save_to_file(inventory_file_path)
                except libvirt_utils.libvirt.libvirtError:
                    print(f"failed to delete node '{node.name}'")
            libvirt_utils.close_connection(conn)

    # Destroy Prime Intellect nodes
    if prime_intellect_nodes:
        pi_client = prime_intellect.PrimeIntellectClient()
        for node in prime_intellect_nodes:
            pod_id = node.vars.get("pod_id")
            if not pod_id:
                print(
                    f"Warning: Node '{node.name}' has no pod_id, "
                    "cannot delete from Prime Intellect"
                )
                # Still remove from inventory
                inv.remove_nodes(lambda inv_node: inv_node.name == node.name)
                inv.save_to_file(inventory_file_path)
                continue

            print(f"Deleting Prime Intellect pod '{node.name}' ({pod_id})...")
            if pi_client.delete_pod(pod_id):
                inv.remove_nodes(lambda inv_node: inv_node.name == node.name)
                print(f"Deleted node '{node.name}'")
                inv.save_to_file(inventory_file_path)
            else:
                print(f"Failed to delete pod '{node.name}' ({pod_id})")


def deploy_libvirt_node(
    conn,
    node: config.Node,
    inv: inventory.NodeInventory,
    run_uuid: uuid.UUID,
    inventory_file_path: str,
) -> bool:
    """Deploy a single libvirt node. Returns True if a new node was created."""
    if libvirt_utils is None:
        raise RuntimeError("libvirt support not enabled. Set ENABLE_LIBVIRT=1 to enable.")
    if not isinstance(node.deployment, config.LibvirtDeployment):
        raise RuntimeError("Expected LibvirtDeployment")

    libvirt_deployment_info: config.LibvirtDeployment = node.deployment

    new_inventory_node = inventory.NodeInput(name=node.name)
    groups = [node.type_name]

    match inv.check_node(new_inventory_node, groups):
        case inventory.NodeStatus.MatchingExists, existing_node:
            print(f"Skipping already existing node '{node.name}'")
            return False
        case inventory.NodeStatus.MismatchingExists, existing_node:
            print(f"Node '{node.name}' already exists in different groups")
            print(f"Existing groups: {existing_node.groups}")
            print(f"New groups: {groups}")
            return False
        case inventory.NodeStatus.NeedsUpdate, existing_node:
            print(f"Updating node '{node.name}'")
        case inventory.NodeStatus.NoneMatching, None:
            libvirt_utils.create_vm(
                conn,
                vm_storage_path="/opt/aether-vm",
                base_image_name=libvirt_deployment_info.image,
                instance_name=node.name,
                ram=libvirt_deployment_info.ram,
                vcpus=libvirt_deployment_info.vcpus,
                disk_size="10G",
                username="aether",
                root_password="foobar",
                os_variant=libvirt_deployment_info.os_variant,
            )
            print(f"Provisioned libvirt node '{node.name}'")
        case _:
            raise RuntimeError("No matching case")

    # Add provider info to node vars
    new_inventory_node.vars["provider"] = "libvirt"

    inv.add_node(
        guid=str(run_uuid),
        node_input=new_inventory_node,
        group_names=groups,
        add_missing_groups=True,
    )

    return True


def provision_prime_intellect_node(
    client: prime_intellect.PrimeIntellectClient,
    node: config.Node,
    inv: inventory.NodeInventory,
    run_uuid: uuid.UUID,
    verbose: bool = True,
) -> Optional[prime_intellect.ProvisionedPod]:
    """
    Provision a single Prime Intellect node (without waiting for it to become active).
    Returns the ProvisionedPod on success, None on failure.
    """
    if not isinstance(node.deployment, config.PrimeIntellectDeployment):
        raise RuntimeError("Expected PrimeIntellectDeployment")

    pi_deployment: config.PrimeIntellectDeployment = node.deployment

    new_inventory_node = inventory.NodeInput(name=node.name)
    groups = [node.type_name]

    match inv.check_node(new_inventory_node, groups):
        case inventory.NodeStatus.MatchingExists, existing_node:
            print(f"Skipping already existing node '{node.name}'")
            return None
        case inventory.NodeStatus.MismatchingExists, existing_node:
            print(f"Node '{node.name}' already exists in different groups")
            print(f"Existing groups: {existing_node.groups}")
            print(f"New groups: {groups}")
            return None
        case inventory.NodeStatus.NeedsUpdate, existing_node:
            print(f"Node '{node.name}' needs update - skipping for now")
            return None
        case inventory.NodeStatus.NoneMatching, None:
            pass
        case _:
            raise RuntimeError("No matching case")

    # Provision the pod - use explicit cloud_id if specified, otherwise search
    if pi_deployment.cloud_id:
        # Use explicit cloud ID
        pod = client.provision_pod_by_cloud_id(
            name=node.name,
            cloud_id=pi_deployment.cloud_id,
            gpu_count=pi_deployment.gpu_count,
            team_id=pi_deployment.team_id,
            regions=pi_deployment.regions,
            os_prefix=pi_deployment.os_prefix,
            disk_size=pi_deployment.disk_size,
            env_vars=pi_deployment.env_vars,
            verbose=verbose,
        )
    else:
        # Find cloud_ids from sibling nodes in the same group (array)
        # to prioritize provisioning on the same instance type
        preferred_cloud_ids: list[str] = []
        for existing_node in inv.nodes:
            if node.type_name in existing_node.groups:
                sibling_cloud_id = existing_node.vars.get("cloud_id")
                if sibling_cloud_id and sibling_cloud_id not in preferred_cloud_ids:
                    preferred_cloud_ids.append(sibling_cloud_id)

        if preferred_cloud_ids and verbose:
            print(f"Found {len(preferred_cloud_ids)} sibling cloud_id(s) to prioritize: {preferred_cloud_ids}")

        # Search for best instance with fallback on failure
        pod = client.provision_pod_with_fallback(
            name=node.name,
            gpu_type=pi_deployment.gpu_type,
            gpu_count=pi_deployment.gpu_count,
            team_id=pi_deployment.team_id,
            regions=pi_deployment.regions,
            os_prefix=pi_deployment.os_prefix,
            provider_blacklist=pi_deployment.provider_blacklist,
            disk_size=pi_deployment.disk_size,
            env_vars=pi_deployment.env_vars,
            max_attempts=5,
            verbose=verbose,
            preferred_cloud_ids=preferred_cloud_ids if preferred_cloud_ids else None,
        )

    if not pod:
        print(f"Failed to provision pod for '{node.name}'")
        return None

    # Store what we know from the provisioning response
    new_inventory_node.address = pod.ip
    ssh_info = pod.ssh_info
    if ssh_info:
        new_inventory_node.ssh_user = ssh_info.user
        new_inventory_node.ssh_port = ssh_info.port

    # Store provider-specific info in vars
    new_inventory_node.vars["provider"] = "prime_intellect"
    new_inventory_node.vars["pod_id"] = pod.pod_id
    new_inventory_node.vars["cloud_id"] = pod.cloud_id
    new_inventory_node.vars["gpu_type"] = pi_deployment.gpu_type
    new_inventory_node.vars["gpu_count"] = pi_deployment.gpu_count
    new_inventory_node.vars["cloud_provider"] = pod.provider_type

    inv.add_node(
        guid=str(run_uuid),
        node_input=new_inventory_node,
        group_names=groups,
        add_missing_groups=True,
    )

    if verbose:
        print(f"Pod '{node.name}' provisioned (status: {pod.status})")
        print(f"  Pod ID: {pod.pod_id}")

    return pod


def wait_for_prime_intellect_pods(
    client: prime_intellect.PrimeIntellectClient,
    provisioned_pods: Dict[str, str],  # node_name -> pod_id
    inv: inventory.NodeInventory,
    inventory_file_path: str,
    timeout: int = 300,
    poll_interval: int = 10,
    verbose: bool = True,
) -> None:
    """
    Wait for all provisioned pods to become active and update inventory with SSH info.
    """
    import time

    if not provisioned_pods:
        return

    pending_pods = dict(provisioned_pods)  # copy
    start_time = time.time()

    if verbose:
        print(f"Waiting for {len(pending_pods)} pods to become active (timeout: {timeout}s)...")

    while pending_pods:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(
                f"Timeout waiting for pods. Still pending: {list(pending_pods.keys())}",
            )
            return

        pod_ids = list(pending_pods.values())
        statuses = client.get_pod_statuses(pod_ids)

        newly_active = []
        for node_name, pod_id in list(pending_pods.items()):
            status = statuses.get(pod_id)
            if status is None:
                if verbose:
                    print(f"  {node_name}: failed to get status")
                continue

            if verbose:
                print(f"  {node_name}: {status.status} (elapsed: {int(elapsed)}s)")

            if status.is_active:
                newly_active.append((node_name, status))
                del pending_pods[node_name]
            elif status.status in ("FAILED", "TERMINATED", "ERROR"):
                print(f"  {node_name}: entered terminal state {status.status}")
                del pending_pods[node_name]

        # Update inventory for newly active pods
        for node_name, status in newly_active:
            ssh_info = status.ssh_info
            # Capture values for closure
            ip = status.ip
            user = ssh_info.user if ssh_info else None
            port = ssh_info.port if ssh_info else None

            def make_update_fn(ip_val, user_val, port_val):
                def update_fn(
                    node: inventory.InventoryNode,
                ) -> inventory.InventoryNode:
                    new = copy.copy(node)
                    new.address = ip_val
                    if user_val and port_val:
                        new.ssh_user = user_val
                        new.ssh_port = port_val
                    return new
                return update_fn

            inv.update_node(node_name, make_update_fn(ip, user, port))

            if verbose:
                print(f"  {node_name}: ACTIVE!")
                if ssh_info:
                    print(f"    SSH: ssh -p {ssh_info.port} {ssh_info.user}@{status.ip}")

        # Save inventory after each update
        if newly_active:
            inv.save_to_file(inventory_file_path)

        if pending_pods:
            time.sleep(poll_interval)

    if verbose:
        print("All pods are active!")


def status(
    inventory_file_path: str,
    setup_definition_file_path: Optional[str] = None,
    auto_update: bool = False,
    no_update: bool = False,
):
    """
    Show the status of all nodes in the inventory.

    Args:
        inventory_file_path: Path to the inventory file
        setup_definition_file_path: Optional path to setup definition for config comparison
        auto_update: Automatically update inventory without prompting
        no_update: Don't check for or prompt about updates
    """
    inv = inventory.NodeInventory.from_file(inventory_file_path)

    # Load config if provided for comparison
    config_nodes: Dict[str, config.Node] = {}
    config_nodes_by_group: Dict[str, list[str]] = {}  # group -> list of node names
    if setup_definition_file_path:
        try:
            parsed_nodes = list(
                config.parse_nodes(read_toml_config_file(setup_definition_file_path))
            )
            for node in parsed_nodes:
                config_nodes[node.name] = node
                if node.type_name not in config_nodes_by_group:
                    config_nodes_by_group[node.type_name] = []
                config_nodes_by_group[node.type_name].append(node.name)
        except Exception as e:
            print(f"Warning: Could not load setup definition: {e}")

    # Separate nodes by provider
    libvirt_nodes = []
    prime_intellect_nodes = []

    for node in inv.nodes:
        provider = node.vars.get("provider")
        if provider == "prime_intellect":
            prime_intellect_nodes.append(node)
        else:
            libvirt_nodes.append(node)

    # Status for libvirt nodes (placeholder)
    if libvirt_nodes:
        print("\n=== Libvirt Nodes ===")
        print(f"{'NAME':<30} {'ADDRESS':<20} {'GROUPS'}")
        print("-" * 70)
        for node in libvirt_nodes:
            address = node.address or "N/A"
            groups = ", ".join(node.groups) if node.groups else "N/A"
            print(f"{node.name:<30} {address:<20} {groups}")
        print("\n(Libvirt status check not implemented)")

    # Status for Prime Intellect nodes
    nodes_needing_update = []  # (node, pod_status, changes_description)
    unknown_nodes = []  # nodes with UNKNOWN status

    if prime_intellect_nodes:
        print("\n=== Prime Intellect Nodes ===")
        pod_ids = [
            node.vars.get("pod_id")
            for node in prime_intellect_nodes
            if node.vars.get("pod_id")
        ]

        statuses = {}
        pi_client = None
        if pod_ids:
            pi_client = prime_intellect.PrimeIntellectClient()
            statuses = pi_client.get_pod_statuses(pod_ids)

        # Group nodes by their first non-'all' group (array group)
        nodes_by_group: Dict[str, list[inventory.InventoryNode]] = {}
        for node in prime_intellect_nodes:
            # Find the primary group (first non-'all' group)
            primary_group = "ungrouped"
            for g in node.groups:
                if g != "all":
                    primary_group = g
                    break
            if primary_group not in nodes_by_group:
                nodes_by_group[primary_group] = []
            nodes_by_group[primary_group].append(node)

        for group_name in sorted(nodes_by_group.keys()):
            group_nodes = nodes_by_group[group_name]
            print(f"\n  [{group_name}]")
            print(f"  {'NAME':<25} {'STATUS':<15} {'ADDRESS':<20} {'SSH'}")
            print("  " + "-" * 83)

            for node in group_nodes:
                pod_id = node.vars.get("pod_id")

                # If no pod_id, node is definitely unknown
                if not pod_id:
                    status_str = "UNKNOWN (no pod_id)"
                    address = node.address or "N/A"
                    if node.ssh_user and node.ssh_port and node.address:
                        ssh_str = f"ssh -p {node.ssh_port} {node.ssh_user}@{node.address}"
                    else:
                        ssh_str = "N/A"
                    unknown_nodes.append(node)
                    print(f"  {node.name:<25} {status_str:<15} {address:<20} {ssh_str}")
                    continue

                pod_status = statuses.get(pod_id)

                # Check if we have a valid status (not None and not empty/unknown status)
                is_valid_status = (
                    pod_status is not None
                    and pod_status.status
                    and pod_status.status.upper() not in ("UNKNOWN", "")
                )

                if is_valid_status and pod_status is not None:
                    status_str = pod_status.status
                    address = pod_status.ip or node.address or "N/A"
                    ssh_info = pod_status.ssh_info
                    if ssh_info:
                        ssh_str = f"ssh -p {ssh_info.port} {ssh_info.user}@{ssh_info.host}"
                    else:
                        ssh_str = "N/A"

                    # Check for differences between inventory and API status
                    if not no_update:
                        changes = []
                        if pod_status.ip and pod_status.ip != node.address:
                            changes.append(f"address: {node.address} -> {pod_status.ip}")
                        if ssh_info:
                            if ssh_info.user != node.ssh_user:
                                changes.append(f"ssh_user: {node.ssh_user} -> {ssh_info.user}")
                            if ssh_info.port != node.ssh_port:
                                changes.append(f"ssh_port: {node.ssh_port} -> {ssh_info.port}")
                        if changes:
                            nodes_needing_update.append((node, pod_status, changes))
                else:
                    status_str = "UNKNOWN"
                    address = node.address or "N/A"
                    if node.ssh_user and node.ssh_port and node.address:
                        ssh_str = f"ssh -p {node.ssh_port} {node.ssh_user}@{node.address}"
                    else:
                        ssh_str = "N/A"
                    unknown_nodes.append(node)

                print(f"  {node.name:<25} {status_str:<15} {address:<20} {ssh_str}")

    print()

    # Handle nodes needing inventory update
    if nodes_needing_update and not no_update:
        print("\n=== Inventory Updates Available ===")
        print("The following nodes have different information in the API than in the inventory:\n")
        for node, pod_status, changes in nodes_needing_update:
            print(f"  {node.name}:")
            for change in changes:
                print(f"    - {change}")

        if auto_update:
            do_update = True
            print("\nAuto-updating inventory...")
        else:
            response = input("\nWould you like to update the inventory with this information? [y/N] ")
            do_update = response.lower() == "y"

        if do_update:
            for node, pod_status, _ in nodes_needing_update:
                ssh_info = pod_status.ssh_info

                def make_update_fn(ip_val, user_val, port_val):
                    def update_fn(n: inventory.InventoryNode) -> inventory.InventoryNode:
                        new = copy.copy(n)
                        if ip_val:
                            new.address = ip_val
                        if user_val:
                            new.ssh_user = user_val
                        if port_val:
                            new.ssh_port = port_val
                        return new
                    return update_fn

                inv.update_node(
                    node.name,
                    make_update_fn(
                        pod_status.ip,
                        ssh_info.user if ssh_info else None,
                        ssh_info.port if ssh_info else None,
                    ),
                )
                print(f"  Updated: {node.name}")

            inv.save_to_file(inventory_file_path)
            print("Inventory saved.")

    # Handle unknown nodes
    if unknown_nodes:
        print("\n=== Unknown Nodes ===")
        print("The following nodes have UNKNOWN status (not found in API):\n")
        for node in unknown_nodes:
            print(f"  - {node.name}")

        response = input("\nWould you like to remove these nodes from the inventory? [y/N] ")
        if response.lower() == "y":
            for node in unknown_nodes:
                inv.remove_nodes(lambda n, name=node.name: n.name == name)
                print(f"  Removed: {node.name}")

            inv.save_to_file(inventory_file_path)
            print("Inventory saved.")

    # Check for config mismatches if setup definition was provided
    if config_nodes:
        inventory_node_names = {node.name for node in inv.nodes}
        config_node_names = set(config_nodes.keys())

        # Nodes in inventory but not in config
        extra_nodes = inventory_node_names - config_node_names
        # Nodes in config but not in inventory
        missing_nodes = config_node_names - inventory_node_names

        if extra_nodes or missing_nodes:
            print("\n=== Config Mismatches ===")

            if extra_nodes:
                print("\nNodes in inventory but NOT in config:")
                for node_name in sorted(extra_nodes):
                    inv_node = next((n for n in inv.nodes if n.name == node_name), None)
                    if inv_node:
                        groups = ", ".join(inv_node.groups) if inv_node.groups else "N/A"
                        print(f"  - {node_name} (groups: {groups})")
                    else:
                        print(f"  - {node_name}")

            if missing_nodes:
                print("\nNodes in config but NOT in inventory (not yet deployed):")
                for node_name in sorted(missing_nodes):
                    cfg_node = config_nodes.get(node_name)
                    if cfg_node:
                        print(f"  - {node_name} (array/group: {cfg_node.type_name})")
                    else:
                        print(f"  - {node_name}")

            # Show summary by group/array
            print("\n  Summary by array/group:")
            all_groups = set(config_nodes_by_group.keys())
            for inv_node in inv.nodes:
                all_groups.update(inv_node.groups)

            # Exclude 'all' group as it contains all nodes by design
            all_groups.discard("all")

            for group in sorted(all_groups):
                config_in_group = set(config_nodes_by_group.get(group, []))
                inv_in_group = {n.name for n in inv.nodes if group in n.groups}

                deployed = len(config_in_group & inv_in_group)
                total_config = len(config_in_group)
                extra_in_group = len(inv_in_group - config_in_group)

                status_parts = []
                if total_config > 0:
                    status_parts.append(f"{deployed}/{total_config} deployed")
                if extra_in_group > 0:
                    status_parts.append(f"{extra_in_group} extra")

                if status_parts:
                    print(f"    {group}: {', '.join(status_parts)}")


def restore(
    inventory_file_path: str,
    setup_definition_file_path: str,
    auto_restore: bool = False,
):
    """
    Restore inventory from Prime Intellect API by matching pod names to config.

    Args:
        inventory_file_path: Path to the inventory file
        setup_definition_file_path: Path to setup definition for matching
        auto_restore: Automatically restore without prompting
    """
    # Load config to get expected node names
    parsed_nodes = list(
        config.parse_nodes(read_toml_config_file(setup_definition_file_path))
    )
    config_nodes: Dict[str, config.Node] = {node.name: node for node in parsed_nodes}

    print(f"Config expects {len(config_nodes)} node(s):")
    for name in sorted(config_nodes.keys()):
        print(f"  - {name}")

    # Load existing inventory (or create empty)
    inv = inventory.read_inventory_file_if_exists_else_create(inventory_file_path)
    existing_node_names = {node.name for node in inv.nodes}

    print(f"\nInventory has {len(existing_node_names)} node(s)")

    # Fetch all pods from Prime Intellect
    print("\nFetching pods from Prime Intellect API...")
    pi_client = prime_intellect.PrimeIntellectClient()
    all_pods = pi_client.list_all_pods()

    if not all_pods:
        print("No pods found in Prime Intellect API")
        return

    print(f"Found {len(all_pods)} pod(s) in Prime Intellect API:")
    for pod in all_pods:
        print(f"  - '{pod.name}' (id: {pod.pod_id}, status: {pod.status})")
    print()

    # Match pods to config nodes
    matched_pods: list[tuple[prime_intellect.PodInfo, config.Node]] = []
    unmatched_pods: list[prime_intellect.PodInfo] = []

    for pod in all_pods:
        if pod.name in config_nodes:
            if pod.name not in existing_node_names:
                matched_pods.append((pod, config_nodes[pod.name]))
            else:
                print(f"  Skipping '{pod.name}' - already in inventory")
        else:
            unmatched_pods.append(pod)

    # Show matched pods
    if matched_pods:
        print("\n=== Pods matching config (can be restored) ===")
        print(f"{'NAME':<30} {'STATUS':<15} {'POD ID':<35} {'GROUP'}")
        print("-" * 100)
        for pod, cfg_node in matched_pods:
            print(f"{pod.name:<30} {pod.status:<15} {pod.pod_id:<35} {cfg_node.type_name}")

    # Show unmatched pods
    if unmatched_pods:
        print("\n=== Pods NOT in config (cannot be auto-restored) ===")
        print(f"{'NAME':<30} {'STATUS':<15} {'POD ID':<35}")
        print("-" * 80)
        for pod in unmatched_pods:
            print(f"{pod.name:<30} {pod.status:<15} {pod.pod_id:<35}")

    if not matched_pods:
        print("\nNo pods to restore.")
        return

    # Ask to restore
    if auto_restore:
        do_restore = True
        print(f"\nAuto-restoring {len(matched_pods)} pod(s) to inventory...")
    else:
        response = input(f"\nWould you like to restore {len(matched_pods)} pod(s) to inventory? [y/N] ")
        do_restore = response.lower() == "y"

    if not do_restore:
        print("Restore cancelled.")
        return

    # Restore pods to inventory
    run_uuid = uuid.uuid4()

    for pod, cfg_node in matched_pods:
        ssh_info = pod.ssh_info

        node_input = inventory.NodeInput(
            name=pod.name,
            address=pod.ip,
            ssh_user=ssh_info.user if ssh_info else None,
            ssh_port=ssh_info.port if ssh_info else None,
        )

        # Store provider-specific info in vars
        node_input.vars["provider"] = "prime_intellect"
        node_input.vars["pod_id"] = pod.pod_id
        node_input.vars["gpu_type"] = pod.gpu_name
        node_input.vars["gpu_count"] = pod.gpu_count
        node_input.vars["cloud_provider"] = pod.provider_type

        groups = [cfg_node.type_name]

        inv.add_node(
            guid=str(run_uuid),
            node_input=node_input,
            group_names=groups,
            add_missing_groups=True,
        )

        print(f"  Restored: {pod.name}")

    inv.save_to_file(inventory_file_path)
    print(f"\nInventory saved to {inventory_file_path}")


def deploy(inventory_file_path: str, setup_definition_file_path: str):
    nodes = list(
        config.parse_nodes(read_toml_config_file(setup_definition_file_path))
    )

    pprint(nodes)

    inv: inventory.NodeInventory = (
        inventory.read_inventory_file_if_exists_else_create(inventory_file_path)
    )
    run_uuid = uuid.uuid4()

    # Separate nodes by provider
    libvirt_nodes = []
    prime_intellect_nodes = []

    for node in nodes:
        if node.deployment is None:
            print(f"Warning: Node '{node.name}' has no deployment config")
            continue

        if isinstance(node.deployment, config.LibvirtDeployment):
            libvirt_nodes.append(node)
        elif isinstance(node.deployment, config.PrimeIntellectDeployment):
            prime_intellect_nodes.append(node)
        else:
            print(
                f"Warning: Unknown deployment type for node '{node.name}': "
                f"{type(node.deployment)}"
            )

    # Deploy libvirt nodes
    libvirt_node_names = []
    if libvirt_nodes:
        if libvirt_utils is None:
            print("Error: libvirt support not enabled. Set ENABLE_LIBVIRT=1 to enable.")
            print(f"Cannot deploy {len(libvirt_nodes)} libvirt node(s)")
        else:
            conn = libvirt_utils.get_connection()

            for node in libvirt_nodes:
                print(f"Deploying libvirt node '{node.name}'")
                if deploy_libvirt_node(
                    conn, node, inv, run_uuid, inventory_file_path
                ):
                    libvirt_node_names.append(node.name)

            pprint(inv.serialize())
            inv.save_to_file(inventory_file_path)

            # Wait for IPs for newly created libvirt nodes
            if libvirt_node_names:
                ips = libvirt_utils.wait_for_vm_ips(
                    conn,
                    expected_vms=libvirt_node_names,
                    poll_interval=4,
                    timeout=800,
                    verbose=True,
                )

                pprint(ips)

                for node_id, address in ips.items():

                    def update_fn(
                        node: inventory.InventoryNode,
                    ) -> inventory.InventoryNode:
                        new = copy.copy(node)
                        new.address = address
                        return new

                    inv.update_node(node_id, update_fn)

                inv.save_to_file(inventory_file_path)

            libvirt_utils.close_connection(conn)

    # Deploy Prime Intellect nodes
    provisioned_pods: Dict[str, str] = {}  # node_name -> pod_id
    pi_timeout = 300  # default timeout

    if prime_intellect_nodes:
        pi_client = prime_intellect.PrimeIntellectClient()

        # First, provision all nodes
        print("\n=== Provisioning Prime Intellect nodes ===")
        for node in prime_intellect_nodes:
            print(f"\nProvisioning Prime Intellect node '{node.name}'")
            pod = provision_prime_intellect_node(
                pi_client, node, inv, run_uuid, verbose=True
            )
            if pod:
                provisioned_pods[node.name] = pod.pod_id
                # Get timeout from deployment config
                if isinstance(node.deployment, config.PrimeIntellectDeployment):
                    pi_timeout = node.deployment.provision_timeout
            # Save after each node in case of failures
            inv.save_to_file(inventory_file_path)

        # Then, wait for all provisioned pods to become active
        if provisioned_pods:
            print("\n=== Waiting for pods to become active ===")
            wait_for_prime_intellect_pods(
                pi_client,
                provisioned_pods,
                inv,
                inventory_file_path,
                timeout=pi_timeout,
                verbose=True,
            )

    print("\nDeployment complete!")
    pprint(inv.serialize())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Multi-backend deployment tool for managing cloud instances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Actions:
  deploy   Provision nodes defined in the setup definition file.
           Creates new instances and adds them to the inventory.

  destroy  Terminate nodes matching the specified filters.
           Removes instances from cloud provider and inventory.

  status   Show the current status of all nodes in the inventory.
           Optionally compares against config and updates inventory.

  restore  Recover inventory from Prime Intellect API.
           Matches existing pods by name to config and rebuilds inventory.

Examples:
  python deploy.py deploy --setup-definition examples/prime-intellect.toml
  python deploy.py status --setup-definition examples/prime-intellect.toml
  python deploy.py destroy --setup-definition examples/prime-intellect.toml all
  python deploy.py restore --setup-definition examples/prime-intellect.toml
        """,
    )
    arg_parser.add_argument(
        "action",
        choices=["deploy", "destroy", "status", "restore"],
        help="Action to perform (see below for details)",
    )
    arg_parser.add_argument(
        "filters",
        nargs="*",
        type=str,
        help="""
            A node will be destroyed when it matches a node name filter or is part of
            all groups which were specified.
            Filter types (and their syntax):
                <literal_node_name> |
                g=<literal_group_name> |
                depl=<literal_deployment_guid> |
                s=<status> (e.g., s=ACTIVE, s=FAILED, s=TERMINATED, s=UNKNOWN) |
                all: all
        """,
    )
    arg_parser.add_argument(
        "--inventory-file-path",
        type=str,
        default="workdir_deploy/inventory.yaml",
    )
    arg_parser.add_argument("--setup-definition", type=str, required=False)
    arg_parser.add_argument(
        "--auto-update",
        action="store_true",
        help="Automatically update inventory without prompting (for status command)",
    )
    arg_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't check for or prompt about inventory updates (for status command)",
    )
    arg_parser.add_argument(
        "--auto-restore",
        action="store_true",
        help="Automatically restore inventory without prompting (for restore command)",
    )
    args = arg_parser.parse_args()

    if args.action == "deploy":
        if not args.setup_definition:
            print("--setup-definition is required for deploy")
            exit(1)
        deploy(args.inventory_file_path, args.setup_definition)
    elif args.action == "destroy":
        if not args.filters:
            print("At least one filter needed to destroy instances")
            exit(1)
        destroy(args.inventory_file_path, args.filters, args.setup_definition)
    elif args.action == "status":
        status(
            args.inventory_file_path,
            setup_definition_file_path=args.setup_definition,
            auto_update=args.auto_update,
            no_update=args.no_update,
        )
    elif args.action == "restore":
        if not args.setup_definition:
            print("--setup-definition is required for restore")
            exit(1)
        restore(
            args.inventory_file_path,
            args.setup_definition,
            auto_restore=args.auto_restore,
        )
