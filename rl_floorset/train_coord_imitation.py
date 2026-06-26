#!/usr/bin/env python3
"""Continuous-coordinate imitation training for FloorSet placement."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iccad2026_evaluate import get_training_dataloader, get_validation_dataloader

from rl_floorset.adapter import build_instance, target_positions_from_training_solution
from rl_floorset.env import SequentialPlacementEnv
from rl_floorset.model import CoordGNNPolicy

try:
    from my_optimizer import MyOptimizer as HeuristicOptimizer
except Exception:  # pragma: no cover
    HeuristicOptimizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../")
    parser.add_argument("--source", choices=["training", "validation"], default="training")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--teacher", choices=["label", "heuristic"], default="heuristic")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--use-edge-gate", action="store_true")
    parser.add_argument("--canvas-scale", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--student-rollout-prob", type=float, default=0.0)
    parser.add_argument("--degree-loss-weight", type=float, default=0.0)
    parser.add_argument("--area-loss-weight", type=float, default=0.0)
    parser.add_argument("--relative-loss-weight", type=float, default=0.0)
    parser.add_argument("--relative-loss-mode", choices=["connected", "all"], default="connected")
    parser.add_argument("--relative-top-k", type=int, default=8)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--checkpoint", default="rl_floorset/checkpoints/coord_policy.pt")
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


def _fit_canvas_to_targets(env: SequentialPlacementEnv, target_pos: torch.Tensor, margin: float = 1.05) -> None:
    if target_pos.numel() == 0:
        return
    max_x = float((target_pos[:, 0] + target_pos[:, 2]).max().item())
    max_y = float((target_pos[:, 1] + target_pos[:, 3]).max().item())
    env.canvas_width = max(env.canvas_width, max_x * margin, 1.0)
    env.canvas_height = max(env.canvas_height, max_y * margin, 1.0)


def _target_xy(env: SequentialPlacementEnv, target_pos: torch.Tensor, block: int) -> torch.Tensor:
    x = float(target_pos[block, 0]) / max(float(env.canvas_width), 1e-6)
    y = float(target_pos[block, 1]) / max(float(env.canvas_height), 1e-6)
    return torch.tensor([x, y], dtype=torch.float32).clamp(0.0, 1.0)


def _loss_weight(env: SequentialPlacementEnv, block: int, args: argparse.Namespace) -> float:
    weight = 1.0
    if args.degree_loss_weight > 0.0 and env.node_features.shape[1] > 4:
        weight += float(args.degree_loss_weight) * float(env.node_features[block, 4])
    if args.area_loss_weight > 0.0:
        mean_area = float(env.instance.area_targets.clamp_min(1e-9).mean().item())
        block_area = float(env.instance.area_targets[block].clamp_min(1e-9).item())
        weight += float(args.area_loss_weight) * min(block_area / max(mean_area, 1e-9), 5.0)
    return weight


def _connected_weights(
    current: int,
    placed_blocks: list[int],
    b2b_conn: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.zeros((len(placed_blocks),), dtype=dtype, device=device)
    if not placed_blocks or b2b_conn is None or b2b_conn.numel() == 0:
        return weights

    placed_to_pos = {block: pos for pos, block in enumerate(placed_blocks)}

    if b2b_conn.dim() == 2 and b2b_conn.shape[0] > current and b2b_conn.shape[1] > max(placed_blocks):
        idx = torch.tensor(placed_blocks, dtype=torch.long, device=b2b_conn.device)
        matrix_weights = b2b_conn[current, idx].float() + b2b_conn[idx, current].float()
        return matrix_weights.to(device=device, dtype=dtype).clamp_min(0.0)

    edge_rows = b2b_conn
    if edge_rows.dim() != 2:
        edge_rows = edge_rows.reshape(-1, edge_rows.shape[-1])
    if edge_rows.shape[0] == 3 and edge_rows.shape[1] != 3:
        edge_rows = edge_rows.t()

    for row in edge_rows:
        if len(row) < 3:
            continue
        a = int(row[0].item())
        b = int(row[1].item())
        weight = float(row[2].item())
        if weight <= 0.0:
            continue
        if a == current and b in placed_to_pos:
            weights[placed_to_pos[b]] += weight
        elif b == current and a in placed_to_pos:
            weights[placed_to_pos[a]] += weight
    return weights.clamp_min(0.0)


def _relative_loss(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    current: int,
    placed_blocks: list[int],
    placed_pred_history: list[torch.Tensor],
    placed_target_history: list[torch.Tensor],
    b2b_conn: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if not placed_pred_history:
        return torch.tensor(0.0, dtype=pred_xy.dtype, device=pred_xy.device)

    history_preds = torch.stack(placed_pred_history).to(pred_xy.device)
    history_targets = torch.stack(placed_target_history).to(pred_xy.device)
    keep = torch.ones((len(placed_blocks),), dtype=torch.bool, device=pred_xy.device)
    weights = torch.ones((len(placed_blocks),), dtype=pred_xy.dtype, device=pred_xy.device)

    if args.relative_loss_mode == "connected":
        edge_weights = _connected_weights(current, placed_blocks, b2b_conn, pred_xy.device, pred_xy.dtype)
        keep = edge_weights > 0.0
        if not bool(keep.any()):
            return torch.tensor(0.0, dtype=pred_xy.dtype, device=pred_xy.device)
        weights = edge_weights

    if args.relative_top_k > 0 and int(keep.sum().item()) > args.relative_top_k:
        candidate_weights = weights.masked_fill(~keep, -1.0)
        _, top_idx = torch.topk(candidate_weights, k=args.relative_top_k)
        top_keep = torch.zeros_like(keep)
        top_keep[top_idx] = True
        keep = keep & top_keep

    history_preds = history_preds[keep]
    history_targets = history_targets[keep]
    weights = weights[keep].clamp_min(1e-9)
    pred_rel = pred_xy.unsqueeze(0) - history_preds
    target_rel = target_xy.to(pred_xy.device).unsqueeze(0) - history_targets
    per_pair_loss = F.smooth_l1_loss(pred_rel, target_rel, reduction="none").mean(dim=1)
    return (per_pair_loss * weights).sum() / weights.sum()


def main() -> None:
    args = parse_args()
    example_iter_fn = _iter_training_examples if args.source == "training" else _iter_validation_examples

    heuristic = None
    if args.teacher == "heuristic":
        if HeuristicOptimizer is None:
            raise RuntimeError("--teacher heuristic requires importable my_optimizer.py")
        heuristic = HeuristicOptimizer(verbose=False)

    model = None
    optimizer = None
    initialized_from_checkpoint = False

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_steps = 0
        rollout_prob = max(0.0, min(1.0, args.student_rollout_prob * (epoch + 1) / max(args.epochs, 1)))

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
            env = SequentialPlacementEnv(instance, canvas_scale=args.canvas_scale)

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

            if model is None:
                obs = env.observe()
                model = CoordGNNPolicy(
                    node_feature_dim=obs["node_features"].shape[1],
                    hidden_dim=args.hidden_dim,
                    message_layers=args.message_layers,
                    use_edge_gate=bool(args.use_edge_gate),
                )
                if args.init_checkpoint and not initialized_from_checkpoint:
                    init_path = Path(args.init_checkpoint)
                    if init_path.exists():
                        ckpt = torch.load(init_path, map_location="cpu")
                        state = ckpt.get("model_state", ckpt)
                        model.load_state_dict(state, strict=False)
                        print(f"initialized from {init_path}")
                    initialized_from_checkpoint = True
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            assert model is not None
            assert optimizer is not None

            env.reset()
            sample_loss = torch.tensor(0.0)
            sample_steps = 0
            done = False
            placed_blocks = []
            placed_pred_history = []
            placed_target_history = []

            while not done:
                obs = env.observe()
                current = int(obs["current_block"].item())
                if current < 0:
                    break

                pred_xy, _ = model(obs)
                target_xy = _target_xy(env, target_pos, current)
                weight = _loss_weight(env, current, args)
                step_loss = F.smooth_l1_loss(pred_xy, target_xy)

                if args.relative_loss_weight > 0.0:
                    rel_loss = _relative_loss(
                        pred_xy,
                        target_xy,
                        current,
                        placed_blocks,
                        placed_pred_history,
                        placed_target_history,
                        b2b_conn,
                        args,
                    )
                    step_loss = step_loss + float(args.relative_loss_weight) * rel_loss

                sample_loss = sample_loss + weight * step_loss
                placed_blocks.append(current)
                placed_pred_history.append(pred_xy.detach())
                placed_target_history.append(target_xy.detach())

                if random.random() < rollout_prob:
                    step_x = float(pred_xy[0].detach()) * float(env.canvas_width)
                    step_y = float(pred_xy[1].detach()) * float(env.canvas_height)
                else:
                    step_x = float(target_pos[current, 0])
                    step_y = float(target_pos[current, 1])
                _, _, done = env.step_xy(step_x, step_y)
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
        print(f"epoch={epoch + 1} avg_step_loss={avg_loss:.6f} steps={total_steps}")

    if model is not None:
        ckpt_path = Path(args.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_version": "coord_regression_v1",
                "model_state": model.state_dict(),
                "hidden_dim": args.hidden_dim,
                "message_layers": args.message_layers,
                "use_edge_gate": bool(args.use_edge_gate),
                "canvas_scale": args.canvas_scale,
                "node_feature_dim": model.node_encoder[0].in_features,
                "relative_loss_weight": float(args.relative_loss_weight),
                "relative_loss_mode": args.relative_loss_mode,
                "relative_top_k": int(args.relative_top_k),
            },
            ckpt_path,
        )
        print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
