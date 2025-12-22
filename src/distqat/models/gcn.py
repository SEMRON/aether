import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from hivemind.moe.server.layers.custom_experts import register_expert_class
from hivemind.moe.server.layers.dropout import DeterministicDropout


def head_sample_input(batch_size, in_dim: int, num_nodes: int, num_edges: int, hid_dim: int):
    if batch_size != 1:
        raise ValueError(f"GCN expects batch_size=1 for full-batch graph, got batch_size={batch_size}")
    return (
        torch.empty((1, num_nodes, in_dim)),
         torch.empty((1, 2, num_edges), dtype=torch.long),
         torch.empty((1, num_nodes, hid_dim), dtype=torch.long),
    )

@register_expert_class("gcn.full", head_sample_input)
class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = DeterministicDropout(drop_prob=0.5)

    def forward(self, inputs, edge_index, dropout_mask):
        inputs = inputs.squeeze(0)
        edge_index = edge_index.squeeze(0)
        dropout_mask = dropout_mask.squeeze(0)

        x = self.conv1(inputs, edge_index)
        x = self.relu(x)
        x = self.dropout(x, dropout_mask)
        x = self.conv2(x, edge_index)
        x = x.unsqueeze(0)
        return x