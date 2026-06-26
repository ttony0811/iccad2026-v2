#!/usr/bin/env python3
"""Imitation pretraining for the FloorSet sequential placement policy.

This is a practical first step before PPO: train the policy to imitate the
provided FloorSet floorplan labels by predicting the nearest grid cell for each
sequential placement step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iccad2026_evaluate import get_training_dataloader, get_validation_dataloader

from rl_floorset.adapter import build_instance, target_positions_from_training_solution
from rl_floorset.env import SequentialPlacementEnv
from rl_floorset.model import EdgeGNNPolicy

try:
    from my_optimizer import MyOptimizer as HeuristicOptimizer
except Exception:  # pragma: no cover - keeps the script usable without baseline.
    HeuristicOptimizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../")
    parser.add_argument("--source", choices=["training", "validation"], default="training")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--teacher", choices=["label", "heuristic"], default="label")
    parser.add_argument("--grid-cols", type=int, default=32)
    parser.add_argument("--grid-rows", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--checkpoint", default="rl_floorset/checkpoints/imitation_policy.pt")
    return parser.parse_args()


def _target_positions_from_polygons(polygons: torch.Tensor, block_count: int) -> torch.Tensor:
    rows = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) == 0:
            rows.append(torch.tensor([0.0, 0.0, 1.0, 1.0]))
            continue
        xy_min = valid.min(dim=0).values
        xy_max = valid.max(dim=0).values
        rows.append(torch.tensor([xy_min[0], xy_min[1], xy_max[0] - xy_min[0], xy_max[1] - xy_min[1]]))
    return torch.stack(rows).float()


def _iter_training_examples(args: argparse.Namespace):
    dataloader = get_training_dataloader(
        data_path=args.data_path,
        batch_size=1,
        num_samples=args.num_samples,
        shuffle=bool(args.shuffle),
    )
    for batch in dataloader:
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, _, fp_sol, _ = batch
        area_target = area_target.squeeze(0)
        b2b_conn = b2b_conn.squeeze(0)
        p2b_conn = p2b_conn.squeeze(0)
        pins_pos = pins_pos.squeeze(0)
        constraints = constraints.squeeze(0)
        fp_sol = fp_sol.squeeze(0)
        block_count = int((area_target != -1).sum().item())
        target_pos = target_positions_from_training_solution(fp_sol, block_count)
        yield area_target, b2b_conn, p2b_conn, pins_pos, constraints, block_count, target_pos


def _iter_validation_examples(args: argparse.Namespace):
    dataloader = get_validation_dataloader(data_path=args.data_path, batch_size=1)
    yielded = 0
    for batch in dataloader:
        inputs, labels = batch
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        polygons, _ = labels
        area_target = area_target.squeeze(0)
        b2b_conn = b2b_conn.squeeze(0)
        p2b_conn = p2b_conn.squeeze(0)
        pins_pos = pins_pos.squeeze(0)
        constraints = constraints.squeeze(0)
        polygons = polygons.squeeze(0)
        block_count = int((area_target != -1).sum().item())
        target_pos = _target_positions_from_polygons(polygons, block_count)
        yield area_target, b2b_conn, p2b_conn, pins_pos, constraints, block_count, target_pos
        yielded += 1
        if args.num_samples is not None and yielded >= args.num_samples:
            break


def _fit_canvas_to_targets(env: SequentialPlacementEnv, target_pos: torch.Tensor) -> None:
    """Keep teacher targets representable in the discrete action grid."""

    if target_pos.numel() == 0:
        return
    max_x = float((target_pos[:, 0] + target_pos[:, 2]).max().item())
    max_y = float((target_pos[:, 1] + target_pos[:, 3]).max().item())
    env.canvas_width = max(env.canvas_width, max_x * 1.05, 1.0)
    env.canvas_height = max(env.canvas_height, max_y * 1.05, 1.0)


def _nearest_valid_action(env: SequentialPlacementEnv, target_x: float, target_y: float) -> int:
    action = env.xy_to_cell(target_x, target_y)
    mask = env.action_mask()
    if 0 <= action < env.grid_size and bool(mask[action]):
        return action

    valid = torch.nonzero(mask, as_tuple=False).flatten()
    if valid.numel() == 0:
        return 0

    best_action = int(valid[0].item())
    best_dist = float("inf")
    for candidate in valid.tolist():
        x, y = env.cell_to_xy(int(candidate))
        dist = (x - target_x) * (x - target_x) + (y - target_y) * (y - target_y)
        if dist < best_dist:
            best_dist = dist
            best_action = int(candidate)
    return best_action


def _masked_cross_entropy(
    logits: torch.Tensor,
    target_action: int,
    action_mask: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    smoothing = max(0.0, min(float(label_smoothing), 0.2))
    if smoothing <= 0.0:
        return F.cross_entropy(logits.view(1, -1), torch.tensor([target_action], dtype=torch.long))

    valid = action_mask.to(dtype=torch.bool, device=logits.device)
    valid_count = int(valid.sum().item())
    if valid_count <= 1:
        return F.cross_entropy(logits.view(1, -1), torch.tensor([target_action], dtype=torch.long))

    log_probs = F.log_softmax(logits, dim=0)
    target = torch.zeros_like(logits)
    target[valid] = smoothing / float(valid_count - 1)
    target[target_action] = 1.0 - smoothing
    return -(target * log_probs).sum()


def main() -> None:
    args = parse_args()
    example_iter_fn = _iter_training_examples if args.source == "training" else _iter_validation_examples

    model = None
    optimizer = None
    heuristic = None
    if args.teacher == "heuristic":
        if HeuristicOptimizer is None:
            raise RuntimeError("--teacher heuristic requires importable my_optimizer.py")
        heuristic = HeuristicOptimizer(verbose=False)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_steps = 0

        for area_target, b2b_conn, p2b_conn, pins_pos, constraints, block_count, target_pos in example_iter_fn(args):
            instance = build_instance(
                block_count,
                area_target,
                b2b_conn,
                p2b_conn,
                pins_pos,
                constraints,
                target_pos,
            )
            env = SequentialPlacementEnv(instance, grid_cols=args.grid_cols, grid_rows=args.grid_rows)

            if heuristic is not None:
                teacher_positions = heuristic.solve(
                    block_count,
                    area_target,
                    b2b_conn,
                    p2b_conn,
                    pins_pos,
                    constraints,
                    target_pos,
                )
                target_pos = torch.tensor(teacher_positions, dtype=torch.float32)

            _fit_canvas_to_targets(env, target_pos)

            if model is None:
                obs = env.observe()
                model = EdgeGNNPolicy(
                    node_feature_dim=obs["node_features"].shape[1],
                    grid_size=env.grid_size,
                    grid_cols=env.grid_cols,
                    grid_rows=env.grid_rows,
                    hidden_dim=args.hidden_dim,
                    message_layers=args.message_layers,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            assert model is not None
            assert optimizer is not None

            env.reset()
            sample_loss = torch.tensor(0.0)
            sample_steps = 0
            done = False
            while not done:
                obs = env.observe()
                current = int(obs["current_block"].item())
                if current < 0:
                    break
                target_action = _nearest_valid_action(
                    env,
                    float(target_pos[current, 0]),
                    float(target_pos[current, 1]),
                )
                logits, _ = model(obs)
                sample_loss = sample_loss + _masked_cross_entropy(
                    logits,
                    target_action,
                    obs["action_mask"],
                    args.label_smoothing,
                )
                _, _, done = env.step(target_action)
                sample_steps += 1

            if sample_steps == 0:
                continue

            optimizer.zero_grad()
            (sample_loss / sample_steps).backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            total_loss += float(sample_loss.detach().item())
            total_steps += sample_steps

        avg_loss = total_loss / max(total_steps, 1)
        print(f"epoch={epoch + 1} avg_step_ce={avg_loss:.4f} steps={total_steps}")

    if model is not None:
        ckpt_path = Path(args.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_version": "coord_policy_v3_residual",
                "model_state": model.state_dict(),
                "grid_cols": args.grid_cols,
                "grid_rows": args.grid_rows,
                "hidden_dim": args.hidden_dim,
                "message_layers": args.message_layers,
                "node_feature_dim": model.node_encoder[0].in_features,
            },
            ckpt_path,
        )
        print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
