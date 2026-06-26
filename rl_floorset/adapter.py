"""Adapters from official FloorSet tensors to RL-friendly objects."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from .structs import FloorSetInstance, Position


def _trim_edges(edges: torch.Tensor, block_count: int, second_col_is_block: bool) -> torch.Tensor:
    if edges is None or edges.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    if edges.dim() != 2:
        edges = edges.reshape(-1, edges.shape[-1])

    candidates = [edges]
    if edges.shape[0] == 3 and edges.shape[1] != 3:
        candidates.append(edges.t())

    best_rows = []
    best_count = -1
    for candidate in candidates:
        if candidate.shape[1] < 3:
            continue
        rows = []
        for row in candidate:
            a = int(row[0].item())
            b = int(row[1].item())
            w = float(row[2].item())
            block_ok = (0 <= b < block_count) if second_col_is_block else (
                0 <= a < block_count and 0 <= b < block_count
            )
            if block_ok and w > 0:
                rows.append([float(a), float(b), w])
        if len(rows) > best_count:
            best_rows = rows
            best_count = len(rows)

    if best_rows:
        return torch.tensor(best_rows, dtype=torch.float32)
    return torch.empty((0, 3), dtype=torch.float32)

def build_instance(
    block_count: int,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor | None = None,
) -> FloorSetInstance:
    """Build a compact instance from the tensors passed to solve()."""

    area_targets = area_targets[:block_count].detach().clone().float()
    constraints = constraints[:block_count].detach().clone().float()
    pins_pos = pins_pos.detach().clone().float()
    b2b = _trim_edges(b2b_connectivity, block_count, second_col_is_block=False)
    p2b = _trim_edges(p2b_connectivity, block_count, second_col_is_block=True)

    if target_positions is None:
        target_positions = torch.full((block_count, 4), -1.0, dtype=torch.float32)
    else:
        target_positions = target_positions[:block_count].detach().clone().float()

    return FloorSetInstance(
        block_count=block_count,
        area_targets=area_targets,
        b2b_connectivity=b2b,
        p2b_connectivity=p2b,
        pins_pos=pins_pos,
        constraints=constraints,
        target_positions=target_positions,
    )


def target_positions_from_training_solution(
    fp_solution: torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    """Convert training label fp_sol [w, h, x, y] into [x, y, w, h]."""

    fp_solution = fp_solution[:block_count].detach().clone().float()
    return torch.stack(
        [fp_solution[:, 2], fp_solution[:, 3], fp_solution[:, 0], fp_solution[:, 1]],
        dim=1,
    )


def default_block_dimensions(instance: FloorSetInstance) -> torch.Tensor:
    """Return [N, 2] dimensions obeying fixed/preplaced hard dimensions."""

    dims = torch.zeros((instance.block_count, 2), dtype=torch.float32)
    immutable = instance.immutable_mask
    for i in range(instance.block_count):
        if bool(immutable[i]):
            tw = float(instance.target_positions[i, 2])
            th = float(instance.target_positions[i, 3])
            if tw > 0 and th > 0:
                dims[i, 0] = tw
                dims[i, 1] = th
                continue
        side = math.sqrt(max(float(instance.area_targets[i]), 1e-9))
        dims[i, 0] = side
        dims[i, 1] = side
    return dims


def positions_tensor_to_list(positions: torch.Tensor) -> List[Position]:
    return [tuple(float(v) for v in row) for row in positions.detach().cpu()]


def positions_list_to_tensor(positions: List[Position]) -> torch.Tensor:
    return torch.tensor(positions, dtype=torch.float32)
