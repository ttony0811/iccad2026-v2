#!/usr/bin/env python3
"""Hybrid optimizer: heuristic baseline with guarded coord-RL candidates."""

from __future__ import annotations

import math
import os
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
from rl_coord_optimizer_template import CoordRLOptimizer  # noqa: E402

try:
    from shapely.geometry import box
    from shapely.ops import unary_union

    SHAPELY_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAPELY_AVAILABLE = False


Position = Tuple[float, float, float, float]


class MyOptimizer(FloorplanOptimizer):
    """Return the coord-RL placement only when a local proxy beats baseline."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.heuristic = HeuristicOptimizer(verbose=verbose)
        self.rl_optimizers = [
            CoordRLOptimizer(verbose=verbose, checkpoint_path=path)
            for path in _candidate_checkpoint_paths()
        ]

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

        baseline_stats = _placement_stats(baseline, b2b_connectivity, p2b_connectivity, pins_pos, constraints)
        _, best_score = _selection_scores_from_stats(baseline_stats, baseline_stats)
        best = baseline

        for rl in self.rl_optimizers:
            try:
                rl_candidate = rl.solve(
                    block_count,
                    area_targets,
                    b2b_connectivity,
                    p2b_connectivity,
                    pins_pos,
                    constraints,
                    target_positions,
                )
            except Exception:
                continue

            if not _hard_legal(block_count, area_targets, constraints, target_positions, rl_candidate):
                continue

            rl_stats = _placement_stats(rl_candidate, b2b_connectivity, p2b_connectivity, pins_pos, constraints)
            if not _select_rl_by_stats(baseline_stats, rl_stats):
                continue

            _, rl_score = _selection_scores_from_stats(baseline_stats, rl_stats)
            if best is baseline or rl_score < best_score:
                best_score = rl_score
                best = rl_candidate

        return best


def _candidate_checkpoint_paths() -> List[Path]:
    checkpoint_dir = Path(__file__).parent / "rl_floorset" / "checkpoints"
    configured = os.environ.get("FLOORSET_COORD_CHECKPOINTS", "").strip()
    if configured:
        raw_names = [name.strip() for name in configured.replace(";", ",").split(",") if name.strip()]
    else:
        raw_names = [
            "coord_scale3_300.pt",
            "coord_heuristic_300.pt",
            "coord_scale2_100.pt",
        ]

    paths = []
    seen = set()
    for name in raw_names:
        path = Path(name)
        if not path.is_absolute():
            path = checkpoint_dir / path
        resolved = str(path)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        paths.append(path)
    return paths or [checkpoint_dir / "coord_policy.pt"]


def _selection_scores(
    baseline: List[Position],
    candidate: List[Position],
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
) -> Tuple[float, float]:
    return _selection_scores_from_stats(
        _placement_stats(baseline, b2b_connectivity, p2b_connectivity, pins_pos, constraints),
        _placement_stats(candidate, b2b_connectivity, p2b_connectivity, pins_pos, constraints),
    )


def _select_rl_by_stats(
    baseline_stats: Tuple[float, float, float],
    rl_stats: Tuple[float, float, float],
) -> bool:
    baseline_hpwl, baseline_area, baseline_v = baseline_stats
    rl_hpwl, rl_area, rl_v = rl_stats
    hpwl_rel = (rl_hpwl - baseline_hpwl) / max(baseline_hpwl, 1e-6)
    area_rel = (rl_area - baseline_area) / max(baseline_area, 1e-6)
    violation_delta = rl_v - baseline_v

    if violation_delta > 1e-9:
        if not (violation_delta <= 0.04 and hpwl_rel <= -0.08 and area_rel <= -0.04):
            return False

    if violation_delta >= -1e-9 and rl_hpwl > baseline_hpwl * 1.02:
        return False

    baseline_cost, rl_cost = _selection_scores_from_stats(baseline_stats, rl_stats)
    if violation_delta < -1e-9 and rl_cost < baseline_cost * 1.01:
        return True
    return rl_cost < baseline_cost * 0.995


def _placement_stats(
    positions: List[Position],
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
) -> Tuple[float, float, float]:
    hpwl = float(calculate_hpwl_b2b(positions, b2b_connectivity) + calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos))
    area = float(calculate_bbox_area(positions))
    violations = _soft_violation_ratio(positions, constraints)
    return hpwl, area, violations


def _selection_scores_from_stats(
    baseline_stats: Tuple[float, float, float],
    candidate_stats: Tuple[float, float, float],
) -> Tuple[float, float]:
    baseline_hpwl, baseline_area, baseline_v = baseline_stats
    candidate_hpwl, candidate_area, candidate_v = candidate_stats

    def score(hpwl: float, area: float, violations: float) -> float:
        hpwl_rel = (hpwl - baseline_hpwl) / max(baseline_hpwl, 1e-6)
        area_rel = (area - baseline_area) / max(baseline_area, 1e-6)
        quality = max(0.2, 1.0 + 0.5 * (hpwl_rel + area_rel))
        return quality * math.exp(2.0 * violations)

    return score(baseline_hpwl, baseline_area, baseline_v), score(candidate_hpwl, candidate_area, candidate_v)

def _soft_violation_ratio(positions: List[Position], constraints: torch.Tensor) -> float:
    block_count = len(positions)
    if constraints is None or constraints.dim() < 2 or len(constraints) < block_count:
        return 0.0

    c = constraints[:block_count]
    ncols = c.shape[1]
    mib = c[:, 2] if ncols > 2 else torch.zeros(block_count)
    cluster = c[:, 3] if ncols > 3 else torch.zeros(block_count)
    boundary = c[:, 4] if ncols > 4 else torch.zeros(block_count)

    n_soft = int((boundary != 0).sum().item())
    max_mib = int(mib.max().item()) if mib.numel() > 0 else 0
    max_cluster = int(cluster.max().item()) if cluster.numel() > 0 else 0
    for gid in range(1, max_mib + 1):
        n_soft += max(0, int((mib == gid).sum().item()) - 1)
    for gid in range(1, max_cluster + 1):
        n_soft += max(0, int((cluster == gid).sum().item()) - 1)
    if n_soft <= 0:
        return 0.0

    violations = _boundary_violations(positions, boundary)
    for gid in range(1, max_mib + 1):
        members = [i for i in range(block_count) if int(mib[i].item()) == gid]
        shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in members}
        violations += max(0, len(shapes) - 1)
    for gid in range(1, max_cluster + 1):
        members = [i for i in range(block_count) if int(cluster[i].item()) == gid]
        violations += _cluster_violations(positions, members)

    return violations / max(n_soft, 1)


def _cluster_violations(positions: List[Position], members: List[int]) -> int:
    if len(members) <= 1:
        return 0
    if SHAPELY_AVAILABLE:
        group_polys = [box(*_rect_bounds(positions[i])) for i in members]
        union_result = unary_union(group_polys)
        if union_result.geom_type == "MultiPolygon":
            return max(0, len(union_result.geoms) - 1)
        return 0
    return max(0, _touching_components(positions, members) - 1)


def _rect_bounds(rect: Position) -> Tuple[float, float, float, float]:
    x, y, w, h = rect
    return x, y, x + w, y + h


def _boundary_violations(positions: List[Position], boundary: torch.Tensor) -> int:
    if not positions:
        return 0
    min_x = min(x for x, _, _, _ in positions)
    min_y = min(y for _, y, _, _ in positions)
    max_x = max(x + w for x, _, w, _ in positions)
    max_y = max(y + h for _, y, _, h in positions)
    count = 0
    eps = 1e-6
    for i, rect in enumerate(positions):
        code = int(boundary[i].item()) if i < len(boundary) else 0
        if code == 0:
            continue
        x, y, w, h = rect
        touches = {
            1: abs(x - min_x) < eps,
            2: abs(x + w - max_x) < eps,
            4: abs(y + h - max_y) < eps,
            8: abs(y - min_y) < eps,
        }
        if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
            count += 1
    return count


def _touching_components(positions: List[Position], members: List[int]) -> int:
    if not members:
        return 0
    parent = {i: i for i in members}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for idx, a in enumerate(members):
        for b in members[idx + 1:]:
            if _rects_touch(positions[a], positions[b]):
                union(a, b)
    return len({find(i) for i in members})


def _rects_touch(a: Position, b: Position) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    eps = 1e-6
    vertical_overlap = min(ay + ah, by + bh) - max(ay, by)
    horizontal_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    touch_x = abs((ax + aw) - bx) < eps or abs((bx + bw) - ax) < eps
    touch_y = abs((ay + ah) - by) < eps or abs((by + bh) - ay) < eps
    return (touch_x and vertical_overlap > eps) or (touch_y and horizontal_overlap > eps)


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
        return (
            constraints is not None
            and constraints.dim() >= 2
            and i < constraints.shape[0]
            and col < constraints.shape[1]
            and bool(float(constraints[i, col]) != 0.0)
        )

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
