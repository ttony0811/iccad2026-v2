"""Feature construction for FloorSet RL models."""

from __future__ import annotations

import torch

from .adapter import default_block_dimensions
from .structs import FloorSetInstance


def build_graph_features(instance: FloorSetInstance) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return node_features, edge_index, edge_weight for a compact block graph."""

    dims = default_block_dimensions(instance)
    area = instance.area_targets.clamp_min(1e-9)
    sqrt_area = torch.sqrt(area)
    total_area = area.sum().clamp_min(1.0)
    fixed = instance.fixed_mask.float().unsqueeze(1)
    preplaced = instance.preplaced_mask.float().unsqueeze(1)
    mib = instance.constraints[:, 2:3] if instance.constraints.shape[1] > 2 else torch.zeros_like(fixed)
    cluster = instance.constraints[:, 3:4] if instance.constraints.shape[1] > 3 else torch.zeros_like(fixed)
    boundary = instance.constraints[:, 4:5] if instance.constraints.shape[1] > 4 else torch.zeros_like(fixed)

    degree = torch.zeros(instance.block_count, dtype=torch.float32)
    edges = []
    weights = []
    for row in instance.b2b_connectivity:
        a, b, w = int(row[0].item()), int(row[1].item()), float(row[2].item())
        if 0 <= a < instance.block_count and 0 <= b < instance.block_count and w > 0:
            edges.append((a, b))
            edges.append((b, a))
            weights.append(w)
            weights.append(w)
            degree[a] += w
            degree[b] += w

    max_dim = dims.max().clamp_min(1.0)
    max_degree = degree.max().clamp_min(1.0)
    node_features = torch.cat(
        [
            (area / total_area).unsqueeze(1),
            (sqrt_area / max_dim).unsqueeze(1),
            dims / max_dim,
            (degree / max_degree).unsqueeze(1),
            fixed,
            preplaced,
            mib.float(),
            cluster.float(),
            boundary.float(),
        ],
        dim=1,
    )

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        edge_weight = edge_weight / edge_weight.max().clamp_min(1.0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    return node_features, edge_index, edge_weight
