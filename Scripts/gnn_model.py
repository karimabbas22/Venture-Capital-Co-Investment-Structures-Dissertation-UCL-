"""
gnn_model.py — minimal heterogeneous GNN for the bipartite investor-startup
graph.

2-layer HeteroConv wrapping SAGEConv per relation (mean aggregation across
relations targeting the same node type), classification head on startup
node embeddings only. Deliberately avoids attention/transformer complexity
as the default architecture -- see ROBUSTNESS_REFRAMING_PLAN.md, Section B,
for the rationale (feasibility over novelty; this graph is small by GNN
standards, and a transparent baseline is preferred for a master's
dissertation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGNN(torch.nn.Module):
    def __init__(self, edge_types, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr="mean")
        self.conv2 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr="mean")
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                 for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return self.head(x_dict["startup"]).squeeze(-1)


def materialize(model: HeteroGNN, data) -> None:
    """SAGEConv's (-1,-1) lazy input dims require one forward pass before
    the optimizer can be constructed (parameters don't exist until then)."""
    model.eval()
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict)
    model.train()
