"""Proxy reward functions for FloorSet RL experiments."""

from __future__ import annotations

import torch

from .structs import FloorSetInstance


def _centers(positions: torch.Tensor) -> torch.Tensor:
    return positions[:, :2] + 0.5 * positions[:, 2:4]


def hpwl_proxy(instance: FloorSetInstance, positions: torch.Tensor) -> torch.Tensor:
    centers = _centers(positions)
    cost = torch.tensor(0.0, dtype=torch.float32, device=positions.device)

    for row in instance.b2b_connectivity.to(positions.device):
        a, b, w = int(row[0].item()), int(row[1].item()), float(row[2].item())
        if 0 <= a < instance.block_count and 0 <= b < instance.block_count:
            cost = cost + w * torch.abs(centers[a] - centers[b]).sum()

    for row in instance.p2b_connectivity.to(positions.device):
        pin_idx, block_idx, w = int(row[0].item()), int(row[1].item()), float(row[2].item())
        if 0 <= block_idx < instance.block_count and 0 <= pin_idx < len(instance.pins_pos):
            pin = instance.pins_pos[pin_idx].to(positions.device)
            cost = cost + w * torch.abs(centers[block_idx] - pin).sum()

    return cost


def bbox_area(positions: torch.Tensor) -> torch.Tensor:
    min_xy = positions[:, :2].min(dim=0).values
    max_xy = (positions[:, :2] + positions[:, 2:4]).max(dim=0).values
    span = (max_xy - min_xy).clamp_min(1e-6)
    return span[0] * span[1]


def overlap_area(positions: torch.Tensor) -> torch.Tensor:
    total = torch.tensor(0.0, dtype=torch.float32, device=positions.device)
    n = positions.shape[0]
    for i in range(n):
        ax1, ay1, aw, ah = positions[i]
        ax2, ay2 = ax1 + aw, ay1 + ah
        for j in range(i + 1, n):
            bx1, by1, bw, bh = positions[j]
            bx2, by2 = bx1 + bw, by1 + bh
            ox = torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)
            oy = torch.minimum(ay2, by2) - torch.maximum(ay1, by1)
            total = total + ox.clamp_min(0) * oy.clamp_min(0)
    return total


def immutable_penalty(instance: FloorSetInstance, positions: torch.Tensor) -> torch.Tensor:
    penalty = torch.tensor(0.0, dtype=torch.float32, device=positions.device)
    target = instance.target_positions.to(positions.device)
    fixed = instance.fixed_mask.to(positions.device)
    preplaced = instance.preplaced_mask.to(positions.device)

    if fixed.any():
        penalty = penalty + torch.abs(positions[fixed, 2:4] - target[fixed, 2:4]).sum()
    if preplaced.any():
        penalty = penalty + torch.abs(positions[preplaced, :] - target[preplaced, :]).sum()
    return penalty


def proxy_reward(
    instance: FloorSetInstance,
    positions: torch.Tensor,
    hpwl_weight: float = 1.0,
    bbox_weight: float = 0.01,
    overlap_weight: float = 1000.0,
    immutable_weight: float = 1000.0,
) -> torch.Tensor:
    """Return a scalar reward; larger is better."""

    cost = (
        hpwl_weight * hpwl_proxy(instance, positions)
        + bbox_weight * bbox_area(positions)
        + overlap_weight * overlap_area(positions)
        + immutable_weight * immutable_penalty(instance, positions)
    )
    return -cost
