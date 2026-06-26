#!/usr/bin/env python3
"""Small hard-constraint smoke test for the FloorSet legalizer."""

from __future__ import annotations

import torch

from rl_floorset.adapter import build_instance
from rl_floorset.legalization_interface import legalize


def overlaps(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return min(ax + aw, bx + bw) - max(ax, bx) > 1e-7 and min(ay + ah, by + bh) - max(ay, by) > 1e-7


def main() -> None:
    block_count = 5
    area_targets = torch.tensor([4.0, 4.0, 9.0, 4.0, 4.0])
    b2b = torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 2.0], [3.0, 4.0, 1.0]])
    p2b = torch.empty((0, 3))
    pins = torch.empty((0, 2))
    constraints = torch.zeros((block_count, 5))
    constraints[0, 1] = 1.0  # preplaced
    constraints[1, 0] = 1.0  # fixed-size movable
    constraints[3, 3] = 7.0  # cluster group
    constraints[4, 3] = 7.0
    target_positions = torch.full((block_count, 4), -1.0)
    target_positions[0] = torch.tensor([0.0, 0.0, 2.0, 2.0])
    target_positions[1] = torch.tensor([-1.0, -1.0, 1.0, 4.0])

    raw = torch.tensor(
        [
            [10.0, 10.0, 99.0, 99.0],  # must be restored to preplaced target
            [0.0, 0.0, 9.0, 9.0],      # must keep fixed dimensions
            [0.0, 0.0, 100.0, 1.0],    # must be rescaled to area 9
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )

    positions = legalize(build_instance(block_count, area_targets, b2b, p2b, pins, constraints, target_positions), raw)

    assert positions[0] == (0.0, 0.0, 2.0, 2.0)
    assert abs(positions[1][2] - 1.0) < 1e-6 and abs(positions[1][3] - 4.0) < 1e-6
    for i in (2, 3, 4):
        assert abs(positions[i][2] * positions[i][3] - float(area_targets[i])) / float(area_targets[i]) <= 0.01
    for i in range(block_count):
        for j in range(i + 1, block_count):
            assert not overlaps(positions[i], positions[j]), (i, j, positions[i], positions[j])

    print("legalization smoke test passed")


if __name__ == "__main__":
    main()
