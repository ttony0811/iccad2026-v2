#!/usr/bin/env python3
"""Debug hybrid selector decisions for selected validation cases."""

from __future__ import annotations

import sys

import torch

from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from hybrid_coord_optimizer import (
    _candidate_checkpoint_paths,
    _placement_stats,
    _select_rl_by_stats,
    _selection_scores,
    _soft_violation_ratio,
)
from my_optimizer import MyOptimizer as HeuristicOptimizer
from rl_coord_optimizer_template import CoordRLOptimizer


def main() -> None:
    ids = [int(arg) for arg in sys.argv[1:]] or [0, 13, 19, 72, 82]
    evaluator = ContestEvaluator(verbose=False)
    evaluator._load_dataset()
    heuristic = HeuristicOptimizer(verbose=False)
    rl_optimizers = [(path.name, CoordRLOptimizer(verbose=False, checkpoint_path=path)) for path in _candidate_checkpoint_paths()]

    for idx in ids:
        sample = evaluator.dataset[idx]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        block_count = int((area_target != -1).sum().item())
        baseline_metrics, target_pos = evaluator._extract_baseline(
            idx, labels, b2b_conn, p2b_conn, pins_pos, block_count
        )

        opt_target_pos = torch.full((block_count, 4), -1.0)
        if target_pos is not None and constraints is not None:
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

        h_pos = heuristic.solve(block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos)
        h_stats = _placement_stats(h_pos, b2b_conn, p2b_conn, pins_pos, constraints)

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
        for ckpt_name, rl in rl_optimizers:
            r_pos = rl.solve(block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos)
            r_stats = _placement_stats(r_pos, b2b_conn, p2b_conn, pins_pos, constraints)
            h_sel, r_sel = _selection_scores(h_pos, r_pos, b2b_conn, p2b_conn, pins_pos, constraints)
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

            print(
                "case",
                idx,
                "ckpt",
                ckpt_name,
                "select",
                "rl" if _select_rl_by_stats(h_stats, r_stats) else "heuristic",
                "selector",
                (round(h_sel, 6), round(r_sel, 6)),
                "stats",
                tuple(round(v, 6) for v in h_stats),
                tuple(round(v, 6) for v in r_stats),
                "soft",
                (round(_soft_violation_ratio(h_pos, constraints), 6), round(_soft_violation_ratio(r_pos, constraints), 6)),
                "official_cost",
                (round(h_metrics.cost, 6), round(r_metrics.cost, 6)),
                "official_metrics",
                (
                    round(h_metrics.hpwl_gap, 6),
                    round(h_metrics.area_gap, 6),
                    round(h_metrics.violations_relative, 6),
                ),
                (
                    round(r_metrics.hpwl_gap, 6),
                    round(r_metrics.area_gap, 6),
                    round(r_metrics.violations_relative, 6),
                ),
            )


if __name__ == "__main__":
    main()
