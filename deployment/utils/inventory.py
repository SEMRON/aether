import copy
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml


@dataclass
class InventoryNode:
    guid: str
    name: str
    groups: List[str]
    address: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: Optional[int] = None
    vars: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InventoryGroup:
    name: str
    vars: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeInput:
    name: str
    address: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: Optional[int] = None
    vars: Dict[str, Any] = field(default_factory=dict)


class NodeStatus(Enum):
    MatchingExists = auto()
    MismatchingExists = auto()
    NeedsUpdate = auto()
    NoneMatching = auto()


class NodeInventory:
    def __init__(self):
        self.nodes: List[InventoryNode] = []
        self.groups: List[InventoryGroup] = []

    def add_group(self, group: InventoryGroup):
        self.groups.append(group)

    def remove_group(self, group_name: str):
        for group in self.groups:
            if group.name == group_name:
                self.groups.remove(group)
                break

    def check_node(
        self, node_input: NodeInput, group_names: List[str]
    ) -> Tuple[NodeStatus, InventoryNode | None]:
        """
        Check if a node already exists in the inventory.
        If it does, check if the groups and variables are the same.
        If they are different, raise an error.
        If they are the same, update the variables and return False.
        If the node does not exist, return True.
        """
        for existing_node in self.nodes:
            if existing_node.name == node_input.name:
                if set(existing_node.groups) != set(group_names):
                    return (NodeStatus.MismatchingExists, existing_node)
                if existing_node.vars != node_input.vars:
                    return (NodeStatus.NeedsUpdate, existing_node)
                return (NodeStatus.MatchingExists, existing_node)
        return NodeStatus.NoneMatching, None

    def add_node(
        self,
        guid: str,
        node_input: NodeInput,
        group_names: List[str],
        add_missing_groups: bool = False,
    ) -> bool:
        """
        Add a node, if a node already exists, just return false.
        If the variables are different, replace them entirely.
        """
        match self.check_node(node_input, group_names):
            case NodeStatus.MatchingExists, existing_node:
                return False
            case NodeStatus.MatchingExists, existing_node:
                raise RuntimeError(
                    f"Node '{node_input.name}' already exists in different groups\n"
                    + f"Existing groups: {existing_node.groups}\n"
                    + f"New groups: {group_names}"
                )
            case NodeStatus.NeedsUpdate, existing_node:
                if existing_node is None:
                    raise RuntimeError()
                existing_node.vars = node_input.vars
                return True
            case NodeStatus.NoneMatching:
                pass

        for group_name in group_names:
            if not any(g.name == group_name for g in self.groups):
                if not add_missing_groups:
                    raise ValueError(f"Group '{group_name}' does not exist")
                self.groups.append(InventoryGroup(name=group_name))
        node = InventoryNode(
            guid=guid,
            name=node_input.name,
            address=node_input.address,
            ssh_user=node_input.ssh_user,
            ssh_port=node_input.ssh_port,
            vars=node_input.vars,
            groups=group_names,
        )
        self.nodes.append(node)

        return True

    def update_node(
        self, name: str, update_fn: Callable[[InventoryNode], InventoryNode]
    ) -> None:
        for i, node in enumerate(self.nodes):
            if node.name == name:
                self.nodes[i] = update_fn(node)

    def update_group(
        self, name: str, update_fn: Callable[[InventoryGroup], InventoryGroup]
    ) -> None:
        for group in self.groups:
            if group.name == name:
                self.groups.remove(group)
                self.groups.append(update_fn(group))

    def remove_nodes(self, filter_fn: Callable[[InventoryNode], bool]) -> None:
        self.nodes = [node for node in self.nodes if not filter_fn(node)]

    def filtered(
        self, filter_fn: Callable[[InventoryNode], bool]
    ) -> "NodeInventory":
        new_inv = NodeInventory()
        new_inv.groups = copy.deepcopy(self.groups)
        new_inv.nodes = [
            copy.deepcopy(node) for node in self.nodes if filter_fn(node)
        ]
        return new_inv

    def serialize(self) -> Dict[str, Any]:
        inventory = dict()

        # Initialize all groups
        for group in self.groups:
            inventory[group.name] = {"hosts": {}}
            if group.vars:
                inventory[group.name].update({"vars": group.vars})

        # Initialized GUIDs and add nodes
        for node in self.nodes:
            legalized_guid = node.guid.replace("-", "_")
            # if the uuid does not yet exist, initialize it
            if legalized_guid not in inventory:
                inventory[legalized_guid] = {"hosts": {}}
                inventory[legalized_guid].update(
                    {"vars": {"deployment_group_guid": node.guid}}
                )

            # add node to GUID
            inventory[legalized_guid]["hosts"][node.name] = dict()
            if node.vars:
                inventory[legalized_guid]["hosts"][node.name].update(
                    {"vars": node.vars}
                )
            if node.address:
                inventory[legalized_guid]["hosts"][node.name].update(
                    {"ansible_host": node.address}
                )
            if node.ssh_user:
                inventory[legalized_guid]["hosts"][node.name].update(
                    {"ansible_user": node.ssh_user}
                )
            if node.ssh_port:
                inventory[legalized_guid]["hosts"][node.name].update(
                    {"ansible_port": node.ssh_port}
                )

            # add the node as a child to all groups in which it is contained
            for group_name in node.groups:
                inventory[group_name]["hosts"][node.name] = {}

        inventory["all"] = {"hosts": {node.name: dict() for node in self.nodes}}

        return inventory

    def save_to_file(self, filepath: str):
        with open(filepath, "w") as f:
            yaml.dump(self.serialize(), f)

    @classmethod
    def from_file(cls, filepath: str) -> "NodeInventory":
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        inv = cls()

        # Loop through each key, value pair as group, data
        for group, data in data.items():
            # If contains 'vars'>'deployment_group_guid', it's a GUID group
            if data.get("vars") and data.get("vars").get(
                "deployment_group_guid"
            ):
                guid = group.replace("_", "-")
                group_vars = data.get("vars", {})

                # Add nodes for this GUID
                for node_name, node_data in data["hosts"].items():
                    node_vars = node_data.get("vars", {})
                    node_address = node_data.get("ansible_host")
                    node_ssh_user = node_data.get("ansible_user")
                    node_ssh_port = node_data.get("ansible_port")
                    node_input = NodeInput(
                        name=node_name,
                        address=node_address,
                        ssh_user=node_ssh_user,
                        ssh_port=node_ssh_port,
                        vars=node_vars,
                    )
                    inv.add_node(guid, node_input, [], True)

            # Otherwise it's a normal group
            else:
                group_vars = data.get("vars", {})
                inv.add_group(InventoryGroup(name=group, vars=group_vars))

                # Add nodes to group
                for node_name in data.get("hosts", {}).keys():
                    for node in inv.nodes:
                        if node.name == node_name:
                            node.groups.append(group)

        return inv


def read_inventory_file_if_exists_else_create(
    path: str,
) -> NodeInventory:
    try:
        return NodeInventory.from_file(path)
    except FileNotFoundError:
        path_parent = os.path.dirname(path)
        if path_parent != "":
            os.makedirs(path_parent, exist_ok=True)
        inv = NodeInventory()
        return inv
