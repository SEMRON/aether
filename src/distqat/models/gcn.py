import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from hivemind.moe.server.layers.custom_experts import register_expert_class
from hivemind.moe.server.layers.dropout import DeterministicDropout
import torch.nn.functional as F


def head_sample_input(batch_size, in_dim: int, num_nodes: int, num_edges: int, hid_dim: int):
    """
    Sample input for the GCN model with neighbor sampling.

    The inputs simulate a mini-batch from NeighborLoader:
    - inputs: sampled node features (1, num_sampled_nodes, in_dim)
    - edge_index: sampled edges (1, 2, num_sampled_edges)
    - dropout_mask: dropout mask (1, num_sampled_nodes, hid_dim)
    - labels: target node labels (1, batch_size) - only for the first batch_size nodes
    """
    if batch_size != 1:
        raise ValueError(f"GCN expects batch_size=1 for mini-batch graph training, got batch_size={batch_size}")
    return (
        torch.empty((1, num_nodes, in_dim)),                   # node features: float32
        torch.empty((1, 2, num_edges), dtype=torch.long),      # edge_index: int64 (required by PyG)
        torch.empty((1, num_nodes, hid_dim), dtype=torch.int8),  # dropout_mask: int8
        torch.empty((1, num_nodes), dtype=torch.long),         # labels: class indices for target nodes
    )


@register_expert_class("gcn.full", head_sample_input)
class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for node classification with neighbor sampling support.

    When using NeighborLoader, the input is a sampled subgraph where:
    - The first `target_size` nodes are the target (seed) nodes
    - Remaining nodes are sampled neighbors
    - Only target nodes have valid labels for loss computation
    """

    def __init__(self, in_dim: int, hid_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = DeterministicDropout(drop_prob=0.5)

    def forward(self, inputs, edge_index, dropout_mask, labels):
        """
        Forward pass for mini-batch graph training with neighbor sampling.

        Args:
            inputs: Node features (1, num_sampled_nodes, in_dim)
            edge_index: Edge indices (1, 2, num_sampled_edges)
            dropout_mask: Dropout mask (1, num_sampled_nodes, hid_dim)
            labels: Target node labels (1, target_size) - labels for seed nodes only

        Returns:
            Tensor of shape (1, 2) containing [loss, accuracy]
        """
        inputs = inputs.squeeze(0)
        edge_index = edge_index.squeeze(0)
        dropout_mask = dropout_mask.squeeze(0)
        labels = labels.squeeze(0).to(inputs.device)
        target_size = labels.shape[0]

        # GCN forward pass on the sampled subgraph
        x = self.conv1(inputs, edge_index)
        x = self.relu(x)
        x = self.dropout(x, dropout_mask)
        x = self.conv2(x, edge_index)

        # Only compute loss on target nodes (first target_size nodes)
        target_logits = x[:target_size]
        loss = F.cross_entropy(target_logits, labels)

        # Compute accuracy on target nodes
        preds = target_logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()

        out = torch.zeros((2,), device=x.device, dtype=x.dtype)
        out[0] = loss
        out[1] = accuracy
        return out.unsqueeze(0)