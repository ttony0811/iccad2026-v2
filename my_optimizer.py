#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template

USAGE:
  1. Copy: cp optimizer_template.py my_optimizer.py
  2. Replace the B*-tree code with your algorithm
  3. Test: python iccad2026_evaluate.py --evaluate my_optimizer.py

BASELINE: B*-tree Simulated Annealing
  - GUARANTEES: Overlap-free, area constraints satisfied
  - NOT HANDLED: Fixed, preplaced, MIB, cluster, boundary constraints

Your solve() receives:
  - block_count: int
  - area_targets: [n] target area per block
  - b2b_connectivity: [edges, 3] (block_i, block_j, weight)
  - p2b_connectivity: [edges, 3] (pin_idx, block_idx, weight)
  - pins_pos: [n_pins, 2] pin (x, y)
  - constraints: [n, 5] (fixed, preplaced, MIB, cluster, boundary)
  - target_positions: [n, 4] target (x, y, w, h) per block.
      All -1 by default (free). For fixed-shape blocks, w and h are set.
      For preplaced blocks, all four (x, y, w, h) are set.

Your solve() must return:
  - List of (x, y, width, height), exactly block_count tuples
  - Floating-point coordinates allowed
  - Any aspect ratio (w/h) allowed

HARD CONSTRAINTS (violation = Cost 10.0):
  - NO OVERLAPS between blocks
  - AREA: w*h within 1% of area_targets[i] (soft blocks only)
  - DIMENSION IMMUTABILITY: Fixed-shape blocks must use exact (w, h) from
    target_positions; preplaced blocks must use exact (x, y, w, h)

RELAXED CONSTRAINTS:
  - Aspect ratio: Any w/h ratio is valid
  - Fixed outline: Removed (implicitly optimized via p2b HPWL and bbox area)
  - Coordinates: Floating-point allowed
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
)


# =============================================================================
# B*-TREE DATA STRUCTURE
# Replace this entire class if using a different representation
# (Sequence Pair, O-tree, Corner Block List, etc.)
# =============================================================================

class BStarTree:
    """
    B*-tree for overlap-free floorplanning.
    
    Left child: placed to the RIGHT of parent
    Right child: placed ABOVE parent (same x)
    """
    
    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = 0
        self._build_random_tree()
    
    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        
        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]
        
        for i in range(1, self.n):
            block = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                elif self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
            else:
                if self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                elif self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
    
    def _insert_at_leaf(self, block: int, start: int):
        current = start
        while True:
            if random.random() < 0.5:
                if self.left[current] == -1:
                    self.left[current] = block
                    self.parent[block] = current
                    return
                current = self.left[current]
            else:
                if self.right[current] == -1:
                    self.right[current] = block
                    self.parent[block] = current
                    return
                current = self.right[current]
    
    def pack(self) -> List[Tuple[float, float, float, float]]:
        """
        Compute (x, y, w, h) from tree structure.
        
        Uses proper contour tracking to ensure overlap-free placement.
        B*-tree rules:
        - Left child: placed to the RIGHT of parent
        - Right child: placed ABOVE parent (same x as parent)
        """
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions
        
        # Contour: sorted list of (x_end, y_top) representing skyline
        # At any x, the contour height is the y_top of the rightmost segment with x_end > x
        contour = [(0.0, 0.0)]  # Start with ground level
        
        def get_contour_y(x_start: float, x_end: float) -> float:
            """Find max y in contour for range [x_start, x_end]."""
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                # Get x_start of this segment
                cx_start = contour[i-1][0] if i > 0 else 0.0
                # Check if segments overlap
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y
        
        def update_contour(x_start: float, x_end: float, y_top: float):
            """Add a new block to the contour."""
            nonlocal contour
            new_contour = []
            
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                
                # Before the new block
                if cx_end <= x_start:
                    new_contour.append((cx_end, cy_top))
                # After the new block
                elif cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                # Overlapping - need to split
                else:
                    # Part before new block
                    if cx_start < x_start:
                        new_contour.append((x_start, cy_top))
                    # Part after new block
                    if cx_end > x_end:
                        new_contour.append((cx_end, cy_top))
            
            # Add the new block segment
            # Find where to insert
            insert_pos = 0
            for i, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = i + 1
            new_contour.insert(insert_pos, (x_end, y_top))
            
            # Sort by x_end and merge adjacent segments with same y
            new_contour.sort(key=lambda x: x[0])
            
            # Merge adjacent segments with same height
            merged = []
            for x_end, y_top in new_contour:
                if merged and merged[-1][1] == y_top:
                    merged[-1] = (x_end, y_top)  # Extend previous
                else:
                    merged.append((x_end, y_top))
            
            contour = merged if merged else [(x_end, 0.0)]
        
        # DFS traversal to place blocks
        def dfs(node: int, parent_right_edge: float):
            if node == -1:
                return
            
            w, h = self.widths[node], self.heights[node]
            
            if node == self.root:
                x = 0.0
                y = 0.0
            else:
                x = parent_right_edge
                y = get_contour_y(x, x + w)
            
            positions[node] = (x, y, w, h)
            update_contour(x, x + w, y + h)
            
            # Left child: to the RIGHT of this node
            dfs(self.left[node], x + w)
            # Right child: ABOVE this node (same x, will stack due to contour)
            dfs(self.right[node], x)
        
        dfs(self.root, 0.0)
        
        # Verify no overlaps (should never happen with correct contour)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_x > 1e-6 and overlap_y > 1e-6:
                    # Fix by pushing j up
                    positions[j] = (x2, max(y1 + h1, y2), w2, h2)
        
        return positions
    
    def copy(self) -> 'BStarTree':
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        return new
    
    # SA moves
    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]
    
    def move_swap(self, b1: int, b2: int):
        """Swap two blocks' dimensions."""
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]
    
    def move_delete_insert(self, block: int):
        """Delete and reinsert block at random position."""
        if self.n <= 1:
            return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block:
            target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h
    
    def _delete_node(self, node: int):
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]
        
        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost
        
        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement
        
        if replacement != -1:
            self.parent[replacement] = parent
        
        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1
    
    def _insert_node(self, node: int, target: int, as_left: bool):
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
        else:
            old_child = self.right[target]
            self.right[target] = node
        self.parent[node] = target
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node


# =============================================================================
# OPTIMIZER CLASS - Replace this with your algorithm
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    B*-tree Simulated Annealing baseline.
    
    REPLACE THIS CLASS WITH YOUR ALGORITHM.
    Keep the solve() signature the same.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.initial_temp = 100.0
        self.final_temp = 1.0
        self.cooling_rate = 0.9
        self.moves_per_temp = 20
    
        """
        B*-tree SA optimization.
        
        REPLACE THIS METHOD with your algorithm.
        Must return List[(x, y, w, h)] with exactly block_count entries.
        """
    def solve(
            self,
            block_count: int,
            area_targets: torch.Tensor,
            b2b_connectivity: torch.Tensor,
            p2b_connectivity: torch.Tensor,
            pins_pos: torch.Tensor,
            constraints: torch.Tensor,
            target_positions: torch.Tensor = None
        ) -> List[Tuple[float, float, float, float]]:
            eps = 1e-7
            positions: List[Tuple[float, float, float, float] | None] = [None] * block_count

            def flag(i: int, col: int) -> bool:
                if constraints is None or constraints.dim() < 2:
                    return False
                if i >= constraints.shape[0] or col >= constraints.shape[1]:
                    return False
                return bool(float(constraints[i, col]) != 0.0)

            def target(i: int, col: int) -> float:
                if target_positions is None:
                    return -1.0
                if i >= target_positions.shape[0] or col >= target_positions.shape[1]:
                    return -1.0
                return float(target_positions[i, col])

            def block_size(i: int) -> Tuple[float, float]:
                if flag(i, 0) or flag(i, 1):
                    tw, th = target(i, 2), target(i, 3)
                    if tw > 0.0 and th > 0.0:
                        return tw, th
                area = max(float(area_targets[i]), eps)
                side = math.sqrt(area)
                return side, side

            sizes = [block_size(i) for i in range(block_count)]

            def group_id(i: int, col: int) -> int:
                if constraints is None or constraints.dim() < 2:
                    return 0
                if i >= constraints.shape[0] or col >= constraints.shape[1]:
                    return 0
                return int(float(constraints[i, col]))

            # MIB groups prefer identical dimensions.  Only force a common
            # shape when doing so still preserves the 1% soft-block area rule.
            mib_groups = {}
            for i in range(block_count):
                gid = group_id(i, 2)
                if gid > 0:
                    mib_groups.setdefault(gid, []).append(i)

            for members in mib_groups.values():
                ref_w = ref_h = None
                for i in members:
                    if flag(i, 0) or flag(i, 1):
                        tw, th = target(i, 2), target(i, 3)
                        if tw > 0.0 and th > 0.0:
                            ref_w, ref_h = tw, th
                            break
                if ref_w is None:
                    avg_area = sum(float(area_targets[i]) for i in members) / max(len(members), 1)
                    ref_w = ref_h = math.sqrt(max(avg_area, eps))

                ref_area = ref_w * ref_h
                safe = True
                for i in members:
                    if flag(i, 0) or flag(i, 1):
                        tw, th = target(i, 2), target(i, 3)
                        safe = safe and abs(tw - ref_w) <= 1e-4 and abs(th - ref_h) <= 1e-4
                    else:
                        area = max(float(area_targets[i]), eps)
                        safe = safe and abs(ref_area - area) / area <= 0.01
                if safe:
                    for i in members:
                        sizes[i] = (ref_w, ref_h)

            for i in range(block_count):
                if flag(i, 1):
                    x, y = target(i, 0), target(i, 1)
                    w, h = sizes[i]
                    positions[i] = (x, y, w, h)

            def rects() -> List[Tuple[int, Tuple[float, float, float, float]]]:
                return [(i, p) for i, p in enumerate(positions) if p is not None]

            def overlaps(
                a: Tuple[float, float, float, float],
                b: Tuple[float, float, float, float],
            ) -> bool:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                return (
                    min(ax + aw, bx + bw) - max(ax, bx) > eps
                    and min(ay + ah, by + bh) - max(ay, by) > eps
                )

            def can_place_many(candidate_rects: List[Tuple[float, float, float, float]]) -> bool:
                for rect in candidate_rects:
                    for _, other in rects():
                        if overlaps(rect, other):
                            return False
                for a_idx in range(len(candidate_rects)):
                    for b_idx in range(a_idx + 1, len(candidate_rects)):
                        if overlaps(candidate_rects[a_idx], candidate_rects[b_idx]):
                            return False
                return True

            def can_place(rect: Tuple[float, float, float, float]) -> bool:
                return can_place_many([rect])

            def can_move_block(i: int, rect: Tuple[float, float, float, float]) -> bool:
                for j, other in rects():
                    if j != i and overlaps(rect, other):
                        return False
                return True

            b2b_edges = []
            for edge in b2b_connectivity:
                if len(edge) >= 3:
                    a, b, weight = int(edge[0]), int(edge[1]), float(edge[2])
                    if 0 <= a < block_count and 0 <= b < block_count and weight > 0:
                        b2b_edges.append((a, b, weight))

            p2b_edges = []
            for edge in p2b_connectivity:
                if len(edge) >= 3:
                    pin_idx, block_idx, weight = int(edge[0]), int(edge[1]), float(edge[2])
                    if (
                        0 <= block_idx < block_count
                        and 0 <= pin_idx < len(pins_pos)
                        and weight > 0
                    ):
                        px, py = float(pins_pos[pin_idx, 0]), float(pins_pos[pin_idx, 1])
                        p2b_edges.append((pin_idx, block_idx, weight, px, py))

            degree = [0.0] * block_count
            for a, b, weight in b2b_edges:
                degree[a] += weight
                degree[b] += weight
            for _, block_idx, weight, _, _ in p2b_edges:
                degree[block_idx] += weight

            adjacency = [{} for _ in range(block_count)]
            for a, b, weight in b2b_edges:
                adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
                adjacency[b][a] = adjacency[b].get(a, 0.0) + weight

            def bbox_area_with_many(candidate_rects: List[Tuple[float, float, float, float]]) -> float:
                all_rects = [p for _, p in rects()] + candidate_rects
                min_x = min(x for x, _, _, _ in all_rects)
                min_y = min(y for _, y, _, _ in all_rects)
                max_x = max(x + w for x, _, w, _ in all_rects)
                max_y = max(y + h for _, y, _, h in all_rects)
                return max((max_x - min_x) * (max_y - min_y), eps)

            def bbox_area_with(rect: Tuple[float, float, float, float]) -> float:
                return bbox_area_with_many([rect])

            def wire_estimate(i: int, rect: Tuple[float, float, float, float]) -> float:
                x, y, w, h = rect
                cx, cy = x + 0.5 * w, y + 0.5 * h
                score = 0.0
                for a, b, weight in b2b_edges:
                    other_idx = b if a == i else a if b == i else -1
                    if other_idx < 0:
                        continue
                    other = positions[other_idx]
                    if other is None:
                        continue
                    ox, oy, ow, oh = other
                    score += weight * (abs(cx - (ox + 0.5 * ow)) + abs(cy - (oy + 0.5 * oh)))
                for _, block_idx, weight, px, py in p2b_edges:
                    if block_idx == i:
                        score += weight * (abs(cx - px) + abs(cy - py))
                return score

            def boundary_code(i: int) -> int:
                return group_id(i, 4)

            def item_wire_estimate(item, item_rects: List[Tuple[float, float, float, float]]) -> float:
                return sum(wire_estimate(block, rect) for block, rect in zip(item["blocks"], item_rects))

            def boundary_bias(item, item_rects: List[Tuple[float, float, float, float]]) -> float:
                score = 0.0
                for block, rect in zip(item["blocks"], item_rects):
                    x, y, w, h = rect
                    code = boundary_code(block)
                    if code & 1:
                        score += 250.0 * abs(x)
                    if code & 8:
                        score += 250.0 * abs(y)
                    if code & 2:
                        score -= 25.0 * (x + w)
                    if code & 4:
                        score -= 25.0 * (y + h)
                return score

            def candidate_xs() -> List[float]:
                xs = {0.0}
                for _, (x, y, w, h) in rects():
                    xs.add(x)
                    xs.add(x + w)
                return sorted(xs)

            def candidate_xs_for_item(item) -> List[float]:
                xs = set(candidate_xs())
                placed_rects = [p for _, p in rects()]
                if placed_rects:
                    min_x = min(x for x, _, _, _ in placed_rects)
                    max_x = max(x + w for x, _, w, _ in placed_rects)
                else:
                    min_x = 0.0
                    max_x = 0.0

                weighted_origin = []
                for block, (rx, _, w, _) in zip(item["blocks"], item["rel"]):
                    code = boundary_code(block)
                    if code & 1:
                        xs.add(min_x - rx)
                    if code & 2:
                        xs.add(max_x - rx - w)

                    for a, b, weight in b2b_edges:
                        other_idx = b if a == block else a if b == block else -1
                        if other_idx < 0 or positions[other_idx] is None:
                            continue
                        ox, oy, ow, oh = positions[other_idx]
                        desired = ox + 0.5 * ow - rx - 0.5 * w
                        weighted_origin.append((desired, weight))

                    for _, block_idx, weight, px, py in p2b_edges:
                        if block_idx == block:
                            desired = px - rx - 0.5 * w
                            weighted_origin.append((desired, weight))

                if weighted_origin:
                    total_w = sum(weight for _, weight in weighted_origin)
                    if total_w > 0.0:
                        center_x = sum(x * weight for x, weight in weighted_origin) / total_w
                        xs.add(center_x)
                        xs.add(center_x - 0.5 * item["width"])
                        xs.add(center_x + 0.5 * item["width"])

                return sorted(xs)

            def lowest_y_for_item(item, x: float) -> float:
                y = 0.0
                while True:
                    bumped = False
                    probes = item_rects_at(item, x, y)
                    for probe_idx, probe in enumerate(probes):
                        rel_y = item["rel"][probe_idx][1]
                        for _, other in rects():
                            if overlaps(probe, other):
                                _, oy, _, oh = other
                                y = max(y, oy + oh - rel_y)
                                bumped = True
                                break
                        if bumped:
                            bumped = True
                            break
                    if not bumped:
                        return y

            def lowest_y_for_x(x: float, w: float, h: float) -> float:
                item = {"blocks": [-1], "rel": [(0.0, 0.0, w, h)], "width": w, "height": h}
                return lowest_y_for_item(item, x)

            def item_rects_at(item, x: float, y: float) -> List[Tuple[float, float, float, float]]:
                return [(x + rx, y + ry, w, h) for rx, ry, w, h in item["rel"]]

            def place_item(item) -> None:
                best_rect = None
                best_item_rects = None
                best_variant = None
                best_score = float("inf")

                for variant in item.get("variants", [item]):
                    for x in candidate_xs():
                        y = lowest_y_for_item(variant, x)
                        item_rects = item_rects_at(variant, x, y)
                        if not can_place_many(item_rects):
                            continue
                        score = (
                            bbox_area_with_many(item_rects)
                            + 0.25 * item_wire_estimate(variant, item_rects)
                            + boundary_bias(variant, item_rects)
                        )
                        if score < best_score:
                            best_score = score
                            best_rect = (x, y, variant["width"], variant["height"])
                            best_item_rects = item_rects
                            best_variant = variant

                if best_rect is None:
                    best_variant = item
                    max_y = 0.0
                    for _, (_, y, _, h2) in rects():
                        max_y = max(max_y, y + h2)
                    best_rect = (0.0, max_y, item["width"], item["height"])
                    best_item_rects = item_rects_at(item, 0.0, max_y)

                for block, rect in zip(best_variant["blocks"], best_item_rects):
                    positions[block] = rect

            def make_item(blocks: List[int]):
                def order_members(members: List[int]) -> List[int]:
                    remaining = set(members)
                    if not remaining:
                        return []

                    first = min(remaining, key=lambda i: (-sizes[i][0] * sizes[i][1], -degree[i], i))
                    ordered_members = [first]
                    remaining.remove(first)

                    while remaining:
                        best = min(
                            remaining,
                            key=lambda i: (
                                -sum(adjacency[i].get(j, 0.0) for j in ordered_members),
                                -sizes[i][0] * sizes[i][1],
                                -degree[i],
                                i,
                            ),
                        )
                        ordered_members.append(best)
                        remaining.remove(best)

                    return ordered_members

                ordered = order_members(blocks)
                has_boundary = any(boundary_code(i) != 0 for i in ordered)

                def build_item(rel: List[Tuple[float, float, float, float]]):
                    if rel:
                        width = max(rx + w for rx, _, w, _ in rel)
                        height = max(ry + h for _, ry, _, h in rel)
                    else:
                        width = 0.0
                        height = 0.0
                    return {
                        "blocks": ordered,
                        "rel": rel,
                        "width": width,
                        "height": height,
                        "area": sum(sizes[i][0] * sizes[i][1] for i in ordered),
                        "degree": sum(degree[i] for i in ordered),
                        "boundary": sum(1 for i in ordered if boundary_code(i) != 0),
                        "fixed": any(flag(i, 0) for i in ordered),
                    }

                def shelf_rel(width_factor: float) -> List[Tuple[float, float, float, float]]:
                    rel = []
                    total_area = sum(sizes[i][0] * sizes[i][1] for i in ordered)
                    max_w = max(sizes[i][0] for i in ordered)
                    target_w = max(max_w, math.sqrt(max(total_area, eps)) * width_factor)
                    cursor_x = 0.0
                    cursor_y = 0.0
                    row_h = 0.0
                    for block in ordered:
                        w, h = sizes[block]
                        if cursor_x > 0.0 and cursor_x + w > target_w:
                            cursor_y += row_h
                            cursor_x = 0.0
                            row_h = 0.0
                        rel.append((cursor_x, cursor_y, w, h))
                        cursor_x += w
                        row_h = max(row_h, h)
                    return rel

                def row_rel() -> List[Tuple[float, float, float, float]]:
                    rel = []
                    cursor_x = 0.0
                    for block in ordered:
                        w, h = sizes[block]
                        rel.append((cursor_x, 0.0, w, h))
                        cursor_x += w
                    return rel

                if len(ordered) >= 4 and not has_boundary:
                    variants = []
                    seen = set()
                    for factor in (0.75, 0.85, 1.0, 1.15):
                        item_variant = build_item(shelf_rel(factor))
                        key = tuple((round(rx, 6), round(ry, 6), round(w, 6), round(h, 6)) for rx, ry, w, h in item_variant["rel"])
                        if key not in seen:
                            seen.add(key)
                            variants.append(item_variant)
                    item = variants[0]
                    item["variants"] = variants
                    return item

                return build_item(row_rel())

            cluster_groups = {}
            assigned = set()
            for i in range(block_count):
                if positions[i] is not None:
                    continue
                gid = group_id(i, 3)
                if gid > 0:
                    cluster_groups.setdefault(gid, []).append(i)

            items = []
            for members in cluster_groups.values():
                movable = [i for i in members if positions[i] is None]
                if len(movable) >= 2:
                    items.append(make_item(movable))
                    assigned.update(movable)

            for i in range(block_count):
                if positions[i] is None and i not in assigned:
                    items.append(make_item([i]))

            def item_phase(item) -> int:
                codes = [boundary_code(i) for i in item["blocks"]]
                if any(code & (1 | 8) for code in codes):
                    return 0
                if any(code & (2 | 4) for code in codes):
                    return 2
                return 1

            items.sort(key=lambda item: (
                item_phase(item),
                not item["fixed"],
                -item["boundary"],
                -item["area"],
                -item["degree"],
            ))

            for item in items:
                place_item(item)

            def layout_boundary_strips() -> None:
                boundary_blocks = [
                    i for i in range(block_count)
                    if positions[i] is not None and boundary_code(i) != 0 and not flag(i, 1)
                ]
                if not boundary_blocks:
                    return

                boundary_set = set(boundary_blocks)
                core_rects = [
                    p for i, p in enumerate(positions)
                    if p is not None and i not in boundary_set
                ]
                if not core_rects:
                    core_rects = [p for p in positions if p is not None]
                if not core_rects:
                    return

                core_min_x = min(x for x, _, _, _ in core_rects)
                core_min_y = min(y for _, y, _, _ in core_rects)
                core_max_x = max(x + w for x, _, w, _ in core_rects)
                core_max_y = max(y + h for _, y, _, h in core_rects)

                corners = {5: [], 6: [], 9: [], 10: []}
                left_side, right_side, top_side, bottom_side = [], [], [], []

                for i in boundary_blocks:
                    code = boundary_code(i)
                    if code in corners:
                        corners[code].append(i)
                    elif code & 2:
                        right_side.append(i)
                    elif code & 4:
                        top_side.append(i)
                    elif code & 8:
                        bottom_side.append(i)
                    elif code & 1:
                        left_side.append(i)

                # Extra corner blocks cannot all satisfy two edges without
                # overlap, so keep the first at the true corner and satisfy the
                # more valuable side for the rest.
                for code, blocks in corners.items():
                    extras = blocks[1:]
                    if not extras:
                        continue
                    del blocks[1:]
                    if code in (6, 10):
                        right_side.extend(extras)
                    elif code == 5:
                        top_side.extend(extras)
                    else:
                        bottom_side.extend(extras)

                def dims_for(blocks: List[int]) -> List[Tuple[float, float]]:
                    return [sizes[i] for i in blocks]

                left_width = max([sizes[i][0] for i in left_side + corners[5] + corners[9]] or [0.0])
                right_width = max([sizes[i][0] for i in right_side + corners[6] + corners[10]] or [0.0])
                top_height = max([sizes[i][1] for i in top_side + corners[5] + corners[6]] or [0.0])
                bottom_height = max([sizes[i][1] for i in bottom_side + corners[9] + corners[10]] or [0.0])

                left_stack_h = sum(h for _, h in dims_for(left_side))
                right_stack_h = sum(h for _, h in dims_for(right_side))
                top_row_w = sum(w for w, _ in dims_for(top_side))
                bottom_row_w = sum(w for w, _ in dims_for(bottom_side))

                left_edge = core_min_x - left_width
                bottom_edge = core_min_y - bottom_height
                top_edge = max(
                    core_max_y + top_height,
                    core_min_y + left_stack_h + top_height,
                    core_min_y + right_stack_h + top_height,
                )
                right_edge = max(
                    core_max_x + right_width,
                    core_min_x + top_row_w + right_width,
                    core_min_x + bottom_row_w + right_width,
                )

                def set_pos(i: int, x: float, y: float) -> None:
                    w, h = sizes[i]
                    positions[i] = (x, y, w, h)

                # Corners first: these define the final bbox corners.
                if corners[5]:
                    i = corners[5][0]
                    set_pos(i, left_edge, top_edge - sizes[i][1])
                if corners[6]:
                    i = corners[6][0]
                    set_pos(i, right_edge - sizes[i][0], top_edge - sizes[i][1])
                if corners[9]:
                    i = corners[9][0]
                    set_pos(i, left_edge, bottom_edge)
                if corners[10]:
                    i = corners[10][0]
                    set_pos(i, right_edge - sizes[i][0], bottom_edge)

                y_cursor = core_min_y
                for i in sorted(left_side, key=lambda b: (-sizes[b][1], -degree[b], b)):
                    set_pos(i, left_edge, y_cursor)
                    y_cursor += sizes[i][1]

                y_cursor = core_min_y
                for i in sorted(right_side, key=lambda b: (-sizes[b][1], -degree[b], b)):
                    set_pos(i, right_edge - sizes[i][0], y_cursor)
                    y_cursor += sizes[i][1]

                x_cursor = core_min_x
                for i in sorted(top_side, key=lambda b: (-sizes[b][0], -degree[b], b)):
                    set_pos(i, x_cursor, top_edge - sizes[i][1])
                    x_cursor += sizes[i][0]

                x_cursor = core_min_x
                for i in sorted(bottom_side, key=lambda b: (-sizes[b][0], -degree[b], b)):
                    set_pos(i, x_cursor, bottom_edge)
                    x_cursor += sizes[i][0]

            def try_boundary_snap() -> None:
                for _ in range(2):
                    placed = [p for p in positions if p is not None]
                    if not placed:
                        return
                    min_x = min(x for x, _, _, _ in placed)
                    min_y = min(y for _, y, _, _ in placed)
                    max_x = max(x + w for x, _, w, _ in placed)
                    max_y = max(y + h for _, y, _, h in placed)
                    for i, rect in enumerate(list(positions)):
                        if rect is None or flag(i, 1):
                            continue
                        code = boundary_code(i)
                        if code == 0:
                            continue
                        x, y, w, h = rect
                        nx, ny = x, y
                        if code & 1:
                            nx = min_x
                        if code & 8:
                            ny = min_y
                        if code & 2:
                            nx = max_x - w
                        if code & 4:
                            ny = max_y - h
                        candidate = (nx, ny, w, h)
                        if can_move_block(i, candidate):
                            positions[i] = candidate

            def boundary_violation_count() -> int:
                placed = [p for p in positions if p is not None]
                if not placed:
                    return 0
                min_x = min(x for x, _, _, _ in placed)
                min_y = min(y for _, y, _, _ in placed)
                max_x = max(x + w for x, _, w, _ in placed)
                max_y = max(y + h for _, y, _, h in placed)
                count = 0
                for i, rect in enumerate(positions):
                    if rect is None:
                        continue
                    code = boundary_code(i)
                    if code == 0:
                        continue
                    x, y, w, h = rect
                    if code & 1 and abs(x - min_x) >= 1e-6:
                        count += 1
                    if code & 2 and abs(x + w - max_x) >= 1e-6:
                        count += 1
                    if code & 4 and abs(y + h - max_y) >= 1e-6:
                        count += 1
                    if code & 8 and abs(y - min_y) >= 1e-6:
                        count += 1
                return count

            def can_move_many_blocks(members: List[int], moved_rects: List[Tuple[float, float, float, float]]) -> bool:
                member_set = set(members)
                for rect_idx, rect in enumerate(moved_rects):
                    for j, other in rects():
                        if j not in member_set and overlaps(rect, other):
                            return False
                    for other_idx in range(rect_idx + 1, len(moved_rects)):
                        if overlaps(rect, moved_rects[other_idx]):
                            return False
                return True

            def try_cluster_boundary_snap() -> None:
                cluster_map = {}
                for i in range(block_count):
                    gid = group_id(i, 3)
                    if gid > 0 and positions[i] is not None:
                        cluster_map.setdefault(gid, []).append(i)

                for _ in range(2):
                    placed = [p for p in positions if p is not None]
                    if not placed:
                        return
                    min_x = min(x for x, _, _, _ in placed)
                    min_y = min(y for _, y, _, _ in placed)
                    max_x = max(x + w for x, _, w, _ in placed)
                    max_y = max(y + h for _, y, _, h in placed)

                    for members in cluster_map.values():
                        if len(members) < 2 or any(flag(i, 1) for i in members):
                            continue
                        if not any(boundary_code(i) != 0 for i in members):
                            continue

                        base_score = boundary_violation_count()
                        best_delta = None
                        best_score = base_score

                        for anchor in members:
                            code = boundary_code(anchor)
                            if code == 0:
                                continue
                            ax, ay, aw, ah = positions[anchor]
                            dx_options = [0.0]
                            dy_options = [0.0]
                            if code & 1:
                                dx_options.append(min_x - ax)
                            if code & 2:
                                dx_options.append(max_x - (ax + aw))
                            if code & 8:
                                dy_options.append(min_y - ay)
                            if code & 4:
                                dy_options.append(max_y - (ay + ah))

                            for dx in dx_options:
                                for dy in dy_options:
                                    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                                        continue
                                    moved = []
                                    for i in members:
                                        x, y, w, h = positions[i]
                                        moved.append((x + dx, y + dy, w, h))
                                    if not can_move_many_blocks(members, moved):
                                        continue

                                    old = [positions[i] for i in members]
                                    for i, rect in zip(members, moved):
                                        positions[i] = rect
                                    score = boundary_violation_count()
                                    for i, rect in zip(members, old):
                                        positions[i] = rect

                                    if score < best_score:
                                        best_score = score
                                        best_delta = (dx, dy)

                        if best_delta is not None:
                            dx, dy = best_delta
                            for i in members:
                                x, y, w, h = positions[i]
                                positions[i] = (x + dx, y + dy, w, h)

            def layout_score() -> float:
                placed = [p for p in positions if p is not None]
                if not placed:
                    return 0.0

                min_x = min(x for x, _, _, _ in placed)
                min_y = min(y for _, y, _, _ in placed)
                max_x = max(x + w for x, _, w, _ in placed)
                max_y = max(y + h for _, y, _, h in placed)
                area = max((max_x - min_x) * (max_y - min_y), eps)

                wire = 0.0
                for a, b, weight in b2b_edges:
                    pa, pb = positions[a], positions[b]
                    if pa is None or pb is None:
                        continue
                    ax, ay, aw, ah = pa
                    bx, by, bw, bh = pb
                    wire += weight * (abs((ax + 0.5 * aw) - (bx + 0.5 * bw)) + abs((ay + 0.5 * ah) - (by + 0.5 * bh)))
                for _, block_idx, weight, px, py in p2b_edges:
                    rect = positions[block_idx]
                    if rect is None:
                        continue
                    x, y, w, h = rect
                    wire += weight * (abs((x + 0.5 * w) - px) + abs((y + 0.5 * h) - py))

                return area + 0.25 * wire + 100.0 * boundary_violation_count()

            def try_single_block_refinement() -> None:
                movable = [
                    i for i in range(block_count)
                    if positions[i] is not None and not flag(i, 1) and group_id(i, 3) == 0
                ]
                movable.sort(key=lambda i: (
                    boundary_code(i) == 0,
                    -degree[i],
                    -sizes[i][0] * sizes[i][1],
                    i,
                ))

                for i in movable[:24]:
                    old = positions[i]
                    if old is None:
                        continue
                    x, y, w, h = old
                    placed = [p for j, p in enumerate(positions) if p is not None and j != i]
                    if not placed:
                        continue

                    min_x = min(px for px, _, _, _ in placed)
                    min_y = min(py for _, py, _, _ in placed)
                    max_x = max(px + pw for px, _, pw, _ in placed)
                    max_y = max(py + ph for _, py, _, ph in placed)

                    xs = {x, min_x, max_x - w, 0.0}
                    ys = {y, min_y, max_y - h, 0.0}
                    x_anchors = [x, min_x, max_x - w, 0.0]
                    y_anchors = [y, min_y, max_y - h, 0.0]

                    weighted_x = []
                    weighted_y = []
                    for a, b, weight in b2b_edges:
                        other_idx = b if a == i else a if b == i else -1
                        if other_idx < 0 or positions[other_idx] is None:
                            continue
                        ox, oy, ow, oh = positions[other_idx]
                        weighted_x.append((ox + 0.5 * ow - 0.5 * w, weight))
                        weighted_y.append((oy + 0.5 * oh - 0.5 * h, weight))
                    for _, block_idx, weight, px, py in p2b_edges:
                        if block_idx == i:
                            weighted_x.append((px - 0.5 * w, weight))
                            weighted_y.append((py - 0.5 * h, weight))

                    if weighted_x:
                        total_w = sum(weight for _, weight in weighted_x)
                        if total_w > 0.0:
                            desired_x = sum(value * weight for value, weight in weighted_x) / total_w
                            xs.add(desired_x)
                            x_anchors.append(desired_x)
                    if weighted_y:
                        total_w = sum(weight for _, weight in weighted_y)
                        if total_w > 0.0:
                            desired_y = sum(value * weight for value, weight in weighted_y) / total_w
                            ys.add(desired_y)
                            y_anchors.append(desired_y)

                    for px, py, pw, ph in placed:
                        xs.add(px)
                        xs.add(px + pw)
                        xs.add(px + pw - w)
                        ys.add(py)
                        ys.add(py + ph)
                        ys.add(py + ph - h)

                    def nearby(values, anchors, limit: int) -> List[float]:
                        ordered = sorted(values, key=lambda value: min(abs(value - anchor) for anchor in anchors))
                        return sorted(ordered[:limit])

                    xs = nearby(xs, x_anchors, 24)
                    ys = nearby(ys, y_anchors, 24)

                    base_score = layout_score()
                    best_score = base_score
                    best_rect = old

                    for nx in xs:
                        for ny in ys:
                            if abs(nx - x) < 1e-9 and abs(ny - y) < 1e-9:
                                continue
                            candidate = (nx, ny, w, h)
                            if not can_move_block(i, candidate):
                                continue
                            positions[i] = candidate
                            score = layout_score()
                            positions[i] = old
                            if score + 1e-9 < best_score:
                                best_score = score
                                best_rect = candidate

                    if best_rect != old:
                        positions[i] = best_rect

            try_cluster_boundary_snap()
            try_boundary_snap()
            try_single_block_refinement()
            try_cluster_boundary_snap()
            try_boundary_snap()

            return [p if p is not None else (0.0, 0.0, *sizes[i]) for i, p in enumerate(positions)]
    
    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """Evaluate solution quality (lower is better)."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01
