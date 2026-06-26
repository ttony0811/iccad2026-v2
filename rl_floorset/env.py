"""A small sequential placement environment for FloorSet instances."""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from .adapter import default_block_dimensions
from .features import build_graph_features
from .reward import proxy_reward
from .structs import FloorSetInstance, PlacementState, Position


class SequentialPlacementEnv:
    """Sequential grid-action environment inspired by AlphaChip.

    This is intentionally lightweight and FloorSet-native. It is not a full
    TF-Agents environment; training code can wrap it for PPO later.
    """

    def __init__(
        self,
        instance: FloorSetInstance,
        grid_cols: int = 32,
        grid_rows: int = 32,
        canvas_scale: float = 1.6,
    ):
        self.instance = instance
        self.grid_cols = int(grid_cols)
        self.grid_rows = int(grid_rows)
        self.grid_size = self.grid_cols * self.grid_rows
        self.canvas_scale = float(canvas_scale)
        self.dims = default_block_dimensions(instance)
        self.canvas_width, self.canvas_height = self._estimate_canvas()
        self.node_features, self.edge_index, self.edge_weight = build_graph_features(instance)
        self.placement_order = self._build_placement_order()
        self.state = self.reset()

    def _estimate_canvas(self) -> tuple[float, float]:
        total_area = float(self.instance.area_targets.clamp_min(0).sum().item())
        side = math.sqrt(max(total_area, 1.0)) * self.canvas_scale
        max_x = side
        max_y = side
        for i in range(self.instance.block_count):
            if bool(self.instance.preplaced_mask[i]):
                x, y, w, h = [float(v) for v in self.instance.target_positions[i]]
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
        return max_x, max_y

    def _build_placement_order(self) -> List[int]:
        movable = [i for i in range(self.instance.block_count) if not bool(self.instance.preplaced_mask[i])]
        degree = torch.zeros(self.instance.block_count)
        for row in self.instance.b2b_connectivity:
            a, b, w = int(row[0].item()), int(row[1].item()), float(row[2].item())
            if 0 <= a < self.instance.block_count and 0 <= b < self.instance.block_count:
                degree[a] += w
                degree[b] += w
        movable.sort(
            key=lambda i: (
                not bool(self.instance.fixed_mask[i]),
                -float(self.dims[i, 0] * self.dims[i, 1]),
                -float(degree[i]),
            )
        )
        return movable

    def reset(self) -> PlacementState:
        positions: List[Position | None] = [None] * self.instance.block_count
        placed_mask = torch.zeros(self.instance.block_count, dtype=torch.bool)
        for i in range(self.instance.block_count):
            if bool(self.instance.preplaced_mask[i]):
                positions[i] = tuple(float(v) for v in self.instance.target_positions[i])
                placed_mask[i] = True
        self.state = PlacementState(positions=positions, placed_mask=placed_mask, step_index=0)
        return self.state

    def current_block(self) -> int | None:
        if self.state.step_index >= len(self.placement_order):
            return None
        return self.placement_order[self.state.step_index]

    def cell_to_xy(self, action: int) -> tuple[float, float]:
        col = int(action) % self.grid_cols
        row = int(action) // self.grid_cols
        cell_w = self.canvas_width / self.grid_cols
        cell_h = self.canvas_height / self.grid_rows
        return col * cell_w, row * cell_h

    def xy_to_cell(self, x: float, y: float) -> int:
        col = min(max(int(x / max(self.canvas_width, 1e-6) * self.grid_cols), 0), self.grid_cols - 1)
        row = min(max(int(y / max(self.canvas_height, 1e-6) * self.grid_rows), 0), self.grid_rows - 1)
        return row * self.grid_cols + col

    def observe(self) -> Dict[str, torch.Tensor]:
        current = self.current_block()
        if current is None:
            current = -1
        return {
            "node_features": self._dynamic_node_features(),
            "edge_index": self.edge_index,
            "edge_weight": self.edge_weight,
            "placed_mask": self.state.placed_mask.clone(),
            "current_block": torch.tensor(current, dtype=torch.long),
            "action_mask": self.action_mask(),
        }

    def _dynamic_node_features(self) -> torch.Tensor:
        dynamic = torch.zeros((self.instance.block_count, 6), dtype=torch.float32)
        canvas_w = max(float(self.canvas_width), 1e-6)
        canvas_h = max(float(self.canvas_height), 1e-6)

        for i, pos in enumerate(self.state.positions):
            if pos is None:
                continue
            x, y, w, h = pos
            dynamic[i, 0] = 1.0
            dynamic[i, 1] = float(x) / canvas_w
            dynamic[i, 2] = float(y) / canvas_h
            dynamic[i, 3] = float(x + w) / canvas_w
            dynamic[i, 4] = float(y + h) / canvas_h

        current = self.current_block()
        if current is not None:
            dynamic[current, 5] = 1.0

        return torch.cat([self.node_features, dynamic], dim=1)

    def action_mask(self) -> torch.Tensor:
        current = self.current_block()
        mask = torch.ones(self.grid_size, dtype=torch.bool)
        if current is None:
            return torch.zeros(self.grid_size, dtype=torch.bool)

        w, h = float(self.dims[current, 0]), float(self.dims[current, 1])
        for action in range(self.grid_size):
            x, y = self.cell_to_xy(action)
            if x + w > self.canvas_width or y + h > self.canvas_height:
                mask[action] = False
        return mask

    def step(self, action: int) -> tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        current = self.current_block()
        if current is None:
            return self.observe(), torch.tensor(0.0), True

        x, y = self.cell_to_xy(int(action))
        return self.step_xy(x, y)

    def step_xy(self, x: float, y: float) -> tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        current = self.current_block()
        if current is None:
            return self.observe(), torch.tensor(0.0), True

        w, h = float(self.dims[current, 0]), float(self.dims[current, 1])
        if bool(self.instance.fixed_mask[current]):
            tw, th = self.instance.target_positions[current, 2:4]
            w, h = float(tw), float(th)
        x = min(max(float(x), 0.0), max(self.canvas_width - w, 0.0))
        y = min(max(float(y), 0.0), max(self.canvas_height - h, 0.0))
        self.state.positions[current] = (x, y, w, h)
        self.state.placed_mask[current] = True
        self.state.step_index += 1

        done = self.state.step_index >= len(self.placement_order)
        reward = torch.tensor(0.0)
        if done:
            reward = proxy_reward(self.instance, self.positions_tensor())
        return self.observe(), reward, done

    def positions_tensor(self) -> torch.Tensor:
        rows = []
        for i, pos in enumerate(self.state.positions):
            if pos is None:
                w, h = float(self.dims[i, 0]), float(self.dims[i, 1])
                rows.append((0.0, 0.0, w, h))
            else:
                rows.append(pos)
        return torch.tensor(rows, dtype=torch.float32)
