#!/usr/bin/env python3
"""Continuous-coordinate RL optimizer template for FloorSet."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import FloorplanOptimizer
from rl_floorset.adapter import build_instance
from rl_floorset.env import SequentialPlacementEnv
from rl_floorset.legalization_interface import legalize
from rl_floorset.model import CoordGNNPolicy


class CoordRLOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False, checkpoint_path: str | Path | None = None):
        super().__init__(verbose)
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent / "rl_floorset" / "checkpoints" / "coord_policy.pt"
        self.checkpoint_path = Path(checkpoint_path)
        self._checkpoint_loaded = False
        self._checkpoint_cache: dict | None = None
        self._model_cache: dict[tuple[int, int, int, bool], CoordGNNPolicy] = {}

    def _load_checkpoint(self) -> dict | None:
        if self._checkpoint_loaded:
            return self._checkpoint_cache
        self._checkpoint_loaded = True
        if not self.checkpoint_path.exists():
            return None
        try:
            self._checkpoint_cache = torch.load(self.checkpoint_path, map_location="cpu")
            return self._checkpoint_cache
        except Exception:
            return None

    def _checkpoint_canvas_scale(self, ckpt: dict | None) -> float:
        if ckpt is None:
            return 1.6
        try:
            return float(ckpt.get("canvas_scale", 1.6))
        except (TypeError, ValueError):
            return 1.6

    def _load_model(self, env: SequentialPlacementEnv, ckpt: dict | None) -> CoordGNNPolicy | None:
        if ckpt is None:
            return None

        observed_dim = int(env.observe()["node_features"].shape[1])
        checkpoint_dim = int(ckpt.get("node_feature_dim", -1))
        if checkpoint_dim != observed_dim:
            return None

        hidden_dim = int(ckpt.get("hidden_dim", 128))
        message_layers = int(ckpt.get("message_layers", 2))
        use_edge_gate = bool(ckpt.get("use_edge_gate", False))
        cache_key = (checkpoint_dim, hidden_dim, message_layers, use_edge_gate)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model = CoordGNNPolicy(
            node_feature_dim=checkpoint_dim,
            hidden_dim=hidden_dim,
            message_layers=message_layers,
            use_edge_gate=use_edge_gate,
        )
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError:
            return None
        model.eval()
        self._model_cache[cache_key] = model
        return model

    def _predict_raw(self, env: SequentialPlacementEnv, ckpt: dict | None) -> torch.Tensor:
        model = self._load_model(env, ckpt)
        env.reset()
        done = False

        while not done:
            obs = env.observe()
            current = int(obs["current_block"].item())
            if current < 0:
                break

            if model is None:
                x, y = 0.0, 0.0
            else:
                with torch.no_grad():
                    xy, _ = model(obs)
                    x = float(xy[0]) * float(env.canvas_width)
                    y = float(xy[1]) * float(env.canvas_height)
            _, _, done = env.step_xy(x, y)

        return env.positions_tensor()

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> List[Tuple[float, float, float, float]]:
        instance = build_instance(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        ckpt = self._load_checkpoint()
        env = SequentialPlacementEnv(instance, canvas_scale=self._checkpoint_canvas_scale(ckpt))
        raw_positions = self._predict_raw(env, ckpt)
        return legalize(instance, raw_positions)


MyOptimizer = CoordRLOptimizer
