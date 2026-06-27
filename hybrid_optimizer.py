#!/usr/bin/env python3
"""Hybrid FloorSet optimizer: heuristic baseline with guarded RL candidates.

This file is experimental. Keep `my_optimizer.py` as the contest-safe baseline
until the RL path consistently improves validation score.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (  # noqa: E402
    FloorplanOptimizer,
    calculate_bbox_area,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
)
from my_optimizer import MyOptimizer as HeuristicOptimizer  # noqa: E402
from rl_optimizer_template import RLOptimizer  # noqa: E402


Position = Tuple[float, float, float, float]


class MyOptimizer(FloorplanOptimizer):
    """Use RL only when its legalized placement beats the heuristic proxy."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.heuristic = HeuristicOptimizer(verbose=verbose)
        self.rl = RLOptimizer(verbose=verbose)

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> List[Position]:
        baseline = self.heuristic.solve(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )

        try:
            rl_candidate = self.rl.solve(
                block_count,
                area_targets,
                b2b_connectivity,
                p2b_connectivity,
                pins_pos,
                constraints,
                target_positions,
            )
        except Exception:
            return baseline

        if not _hard_legal(block_count, area_targets, constraints, target_positions, rl_candidate):
            return baseline

        baseline_cost = _proxy_cost(baseline, b2b_connectivity, p2b_connectivity, pins_pos)
        rl_cost = _proxy_cost(rl_candidate, b2b_connectivity, p2b_connectivity, pins_pos)

        # The evaluator's normalized score is not exactly this proxy, so require
        # a small margin before risking the RL candidate.
        if rl_cost < baseline_cost * 0.98:
            return rl_candidate
        return baseline


def _proxy_cost(
    positions: List[Position],
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
) -> float:
    return float(
        calculate_hpwl_b2b(positions, b2b_connectivity)
        + calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
        + 0.01 * calculate_bbox_area(positions)
    )


def _hard_legal(
    block_count: int,
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor | None,
    positions: List[Position],
) -> bool:
    if len(positions) != block_count:
        return False

    def flag(i: int, col: int) -> bool:
        return constraints is not None and constraints.dim() >= 2 and col < constraints.shape[1] and bool(float(constraints[i, col]) != 0.0)

    for i, rect in enumerate(positions):
        x, y, w, h = rect
        if not all(math.isfinite(v) for v in rect) or w <= 0.0 or h <= 0.0:
            return False

        if target_positions is not None and flag(i, 1):
            target = tuple(float(v) for v in target_positions[i])
            if any(abs(a - b) > 1e-4 for a, b in zip(rect, target)):
                return False
        elif target_positions is not None and flag(i, 0):
            tw, th = float(target_positions[i, 2]), float(target_positions[i, 3])
            if abs(w - tw) > 1e-4 or abs(h - th) > 1e-4:
                return False
        else:
            area = max(float(area_targets[i]), 1e-7)
            if abs(w * h - area) / area > 0.01:
                return False

    for i in range(block_count):
        ax, ay, aw, ah = positions[i]
        for j in range(i + 1, block_count):
            bx, by, bw, bh = positions[j]
            if min(ax + aw, bx + bw) - max(ax, bx) > 1e-7 and min(ay + ah, by + bh) - max(ay, by) > 1e-7:
                return False
    return True
