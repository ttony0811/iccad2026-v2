"""Shared data structures for the FloorSet RL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

Position = Tuple[float, float, float, float]


@dataclass(frozen=True)
class FloorSetInstance:
    """A trimmed, explicit view of one FloorSet placement instance."""

    block_count: int
    area_targets: torch.Tensor
    b2b_connectivity: torch.Tensor
    p2b_connectivity: torch.Tensor
    pins_pos: torch.Tensor
    constraints: torch.Tensor
    target_positions: torch.Tensor

    @property
    def fixed_mask(self) -> torch.Tensor:
        if self.constraints.numel() == 0 or self.constraints.shape[1] < 1:
            return torch.zeros(self.block_count, dtype=torch.bool)
        return self.constraints[: self.block_count, 0] != 0

    @property
    def preplaced_mask(self) -> torch.Tensor:
        if self.constraints.numel() == 0 or self.constraints.shape[1] < 2:
            return torch.zeros(self.block_count, dtype=torch.bool)
        return self.constraints[: self.block_count, 1] != 0

    @property
    def immutable_mask(self) -> torch.Tensor:
        return self.fixed_mask | self.preplaced_mask


@dataclass
class PlacementState:
    """Mutable state used by the sequential RL environment."""

    positions: List[Position | None]
    placed_mask: torch.Tensor
    step_index: int
