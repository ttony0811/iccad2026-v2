#!/usr/bin/env python3
"""Rank coord-RL checkpoints against the heuristic baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from hybrid_coord_optimizer import _placement_stats, _select_rl_by_stats
from my_optimizer import MyOptimizer as HeuristicOptimizer
from rl_coord_optimizer_template import CoordRLOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-glob", default="coord*.pt")
    parser.add_argument("--ids", default="0,13,19,72,82")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--show-wins", action="store_true")
    return parser.parse_args()


def _parse_ids(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _target_positions_for_optimizer(block_count: int, constraints: torch.Tensor, target_pos: torch.Tensor | None) -> torch.Tensor:
    opt_target_pos = torch.full((block_count, 4), -1.0)
    if target_pos is None or constraints is None:
        return opt_target_pos

    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_preplaced = nc > 1 and constraints[i, 1] != 0
        if is_preplaced:
            tx, ty, tw, th = target_pos[i]
            opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_pos[i]
            opt_target_pos[i, 2] = tw
            opt_target_pos[i, 3] = th
    return opt_target_pos


def _iter_cases(evaluator: ContestEvaluator, ids: Iterable[int]):
    for idx in ids:
        sample = evaluator.dataset[idx]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        block_count = int((area_target != -1).sum().item())
        baseline_metrics, target_pos = evaluator._extract_baseline(
            idx, labels, b2b_conn, p2b_conn, pins_pos, block_count
        )
        opt_target_pos = _target_positions_for_optimizer(block_count, constraints, target_pos)
        yield idx, area_target, b2b_conn, p2b_conn, pins_pos, constraints, block_count, baseline_metrics, target_pos, opt_target_pos


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path("rl_floorset/checkpoints")
    checkpoints = sorted(checkpoint_dir.glob(args.checkpoint_glob))
    if not checkpoints:
        raise SystemExit(f"no checkpoints matched {checkpoint_dir / args.checkpoint_glob}")

    evaluator = ContestEvaluator(verbose=False)
    evaluator._load_dataset()
    ids = list(range(len(evaluator.dataset))) if args.all else _parse_ids(args.ids)
    heuristic = HeuristicOptimizer(verbose=False)

    case_cache = []
    for case in _iter_cases(evaluator, ids):
        idx, area_target, b2b_conn, p2b_conn, pins_pos, constraints, block_count, baseline_metrics, target_pos, opt_target_pos = case
        h_pos = heuristic.solve(block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos)
        h_metrics = evaluate_solution(
            {"positions": h_pos, "runtime": 1.0},
            baseline_metrics,
            constraints,
            b2b_conn,
            p2b_conn,
            pins_pos,
            area_target,
            target_pos,
            median_runtime=1.0,
        )
        h_stats = _placement_stats(h_pos, b2b_conn, p2b_conn, pins_pos, constraints)
        case_cache.append((case, h_pos, h_metrics, h_stats))

    print(f"cases={len(case_cache)} checkpoints={len(checkpoints)} glob={args.checkpoint_glob}")
    print(f"heuristic_avg={sum(h.cost for _, _, h, _ in case_cache) / max(len(case_cache), 1):.6f}")

    rows = []
    for ckpt in checkpoints:
        rl = CoordRLOptimizer(verbose=False, checkpoint_path=ckpt)
        costs = []
        wins = []
        selected = []
        selected_bad = []
        failures = 0
        for case, h_pos, h_metrics, h_stats in case_cache:
            idx, area_target, b2b_conn, p2b_conn, pins_pos, constraints, block_count, baseline_metrics, target_pos, opt_target_pos = case
            try:
                r_pos = rl.solve(block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos)
                r_metrics = evaluate_solution(
                    {"positions": r_pos, "runtime": 1.0},
                    baseline_metrics,
                    constraints,
                    b2b_conn,
                    p2b_conn,
                    pins_pos,
                    area_target,
                    target_pos,
                    median_runtime=1.0,
                )
                r_stats = _placement_stats(r_pos, b2b_conn, p2b_conn, pins_pos, constraints)
            except Exception:
                failures += 1
                continue

            costs.append(r_metrics.cost)
            if r_metrics.cost < h_metrics.cost:
                wins.append((idx, h_metrics.cost, r_metrics.cost))
            if _select_rl_by_stats(h_stats, r_stats):
                selected.append((idx, h_metrics.cost, r_metrics.cost))
                if r_metrics.cost >= h_metrics.cost:
                    selected_bad.append((idx, h_metrics.cost, r_metrics.cost))

        avg = sum(costs) / max(len(costs), 1)
        rows.append((avg, -len(wins), len(selected_bad), ckpt.name, wins, selected, selected_bad, failures))

    rows.sort()
    for avg, neg_wins, selected_bad_count, name, wins, selected, selected_bad, failures in rows[: args.top]:
        print(
            f"{name:28s} avg={avg:.6f} wins={-neg_wins:3d} "
            f"selected={len(selected):3d} selected_bad={selected_bad_count:3d} failures={failures:3d}"
        )
        if args.show_wins:
            print("  wins:", [(i, round(h, 4), round(r, 4)) for i, h, r in wins[:20]])
            print("  selected_bad:", [(i, round(h, 4), round(r, 4)) for i, h, r in selected_bad[:20]])


if __name__ == "__main__":
    main()
