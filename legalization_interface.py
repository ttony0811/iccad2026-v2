"""Legalization boundary between the RL placer and the FloorSet evaluator."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from .adapter import default_block_dimensions
from .structs import FloorSetInstance, Position


EPS = 1e-7


def legalize(instance: FloorSetInstance, raw_positions: torch.Tensor | None) -> List[Position]:
    """Repair an RL/raw placement into an evaluator-legal placement.

    The legalizer treats official hard constraints as non-negotiable:
    preplaced blocks stay exactly at their target rectangles, fixed blocks keep
    their target dimensions, soft blocks keep their target area, and the final
    layout has no overlap. It still uses the raw placement as guidance for
    ordering and candidate coordinates, so an RL policy can improve HPWL without
    needing to learn every detail of geometric cleanup.
    """

    dims = _legal_dimensions(instance, raw_positions)
    raw_context = _compact_raw_context(instance, raw_positions, dims)
    positions: List[Position | None] = [None] * instance.block_count

    for i in range(instance.block_count):
        if _is_preplaced(instance, i):
            x, y, w, h = [float(v) for v in instance.target_positions[i]]
            positions[i] = (x, y, w, h)

    adjacency, pin_adjacency, degree = _connectivity(instance)
    items = _build_items(instance, raw_positions, dims, degree, adjacency, positions, raw_context)
    items.sort(
        key=lambda item: (
            item["phase"],
            -item["boundary"],
            -item["area"],
            -item["degree"],
            item["id"],
        )
    )

    for item in items:
        _place_item(instance, item, raw_positions, positions, adjacency, pin_adjacency, dims)

    _cluster_group_polish(instance, raw_positions, raw_context, positions, adjacency, pin_adjacency, dims, limit=12)
    _single_block_polish(instance, raw_positions, positions, adjacency, pin_adjacency, dims, limit=32)
    _snap_boundary_blocks(instance, positions, adjacency, pin_adjacency)
    _cluster_group_polish(instance, raw_positions, raw_context, positions, adjacency, pin_adjacency, dims, limit=8)
    _snap_boundary_blocks(instance, positions, adjacency, pin_adjacency)

    legalized = [
        pos if pos is not None else (0.0, 0.0, float(dims[i, 0]), float(dims[i, 1]))
        for i, pos in enumerate(positions)
    ]
    _assert_hard_legal(instance, legalized)
    return legalized


def conservative_legalize(
    instance: FloorSetInstance,
    raw_positions: torch.Tensor | None = None,
) -> List[Position]:
    """Compatibility wrapper used by older RL template code."""

    return legalize(instance, raw_positions)


def _is_fixed(instance: FloorSetInstance, i: int) -> bool:
    return bool(instance.fixed_mask[i])


def _is_preplaced(instance: FloorSetInstance, i: int) -> bool:
    return bool(instance.preplaced_mask[i])


def _group_id(instance: FloorSetInstance, i: int, col: int) -> int:
    if instance.constraints.numel() == 0 or instance.constraints.shape[1] <= col:
        return 0
    return int(float(instance.constraints[i, col]))


def _boundary_code(instance: FloorSetInstance, i: int) -> int:
    return _group_id(instance, i, 4)


def _raw_rect(raw_positions: torch.Tensor | None, i: int) -> Position | None:
    if raw_positions is None or i >= raw_positions.shape[0] or raw_positions.shape[1] < 4:
        return None
    row = raw_positions[i]
    values = [float(v) for v in row[:4]]
    if not all(math.isfinite(v) for v in values):
        return None
    return tuple(values)  # type: ignore[return-value]


def _legal_dimensions(instance: FloorSetInstance, raw_positions: torch.Tensor | None) -> torch.Tensor:
    dims = default_block_dimensions(instance).clone().float()

    for i in range(instance.block_count):
        if _is_fixed(instance, i) or _is_preplaced(instance, i):
            tw = float(instance.target_positions[i, 2])
            th = float(instance.target_positions[i, 3])
            if tw > 0.0 and th > 0.0:
                dims[i, 0] = tw
                dims[i, 1] = th
            continue

        area = max(float(instance.area_targets[i]), EPS)
        raw = _raw_rect(raw_positions, i)
        if raw is not None and raw[2] > EPS and raw[3] > EPS:
            aspect = max(min(abs(raw[2] / raw[3]), 8.0), 0.125)
            dims[i, 0] = math.sqrt(area * aspect)
            dims[i, 1] = math.sqrt(area / aspect)

    _apply_safe_mib_dimensions(instance, dims)
    return dims


def _apply_safe_mib_dimensions(instance: FloorSetInstance, dims: torch.Tensor) -> None:
    groups: Dict[int, List[int]] = {}
    for i in range(instance.block_count):
        gid = _group_id(instance, i, 2)
        if gid > 0:
            groups.setdefault(gid, []).append(i)

    for members in groups.values():
        if len(members) < 2:
            continue

        ref_w = ref_h = None
        for i in members:
            if _is_fixed(instance, i) or _is_preplaced(instance, i):
                tw = float(instance.target_positions[i, 2])
                th = float(instance.target_positions[i, 3])
                if tw > 0.0 and th > 0.0:
                    ref_w, ref_h = tw, th
                    break

        if ref_w is None:
            avg_area = sum(float(instance.area_targets[i]) for i in members) / len(members)
            avg_aspect = sum(float(dims[i, 0] / max(float(dims[i, 1]), EPS)) for i in members) / len(members)
            avg_aspect = max(min(avg_aspect, 8.0), 0.125)
            ref_w = math.sqrt(max(avg_area, EPS) * avg_aspect)
            ref_h = math.sqrt(max(avg_area, EPS) / avg_aspect)

        ref_area = ref_w * ref_h
        safe = True
        for i in members:
            if _is_fixed(instance, i) or _is_preplaced(instance, i):
                safe = safe and abs(float(dims[i, 0]) - ref_w) <= 1e-4 and abs(float(dims[i, 1]) - ref_h) <= 1e-4
            else:
                area = max(float(instance.area_targets[i]), EPS)
                safe = safe and abs(ref_area - area) / area <= 0.01

        if safe:
            for i in members:
                dims[i, 0] = ref_w
                dims[i, 1] = ref_h


def _connectivity(instance: FloorSetInstance) -> Tuple[List[Dict[int, float]], List[List[Tuple[float, float, float]]], List[float]]:
    adjacency: List[Dict[int, float]] = [{} for _ in range(instance.block_count)]
    pin_adjacency: List[List[Tuple[float, float, float]]] = [[] for _ in range(instance.block_count)]
    degree = [0.0] * instance.block_count

    for row in instance.b2b_connectivity:
        a, b, weight = int(row[0]), int(row[1]), float(row[2])
        if 0 <= a < instance.block_count and 0 <= b < instance.block_count and weight > 0.0:
            adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
            adjacency[b][a] = adjacency[b].get(a, 0.0) + weight
            degree[a] += weight
            degree[b] += weight

    for row in instance.p2b_connectivity:
        pin_idx = int(row[0])
        block = int(row[1])
        weight = float(row[2])
        if 0 <= block < instance.block_count and 0 <= pin_idx < len(instance.pins_pos) and weight > 0.0:
            px, py = float(instance.pins_pos[pin_idx, 0]), float(instance.pins_pos[pin_idx, 1])
            pin_adjacency[block].append((px, py, weight))
            degree[block] += weight

    return adjacency, pin_adjacency, degree


def _build_items(
    instance: FloorSetInstance,
    raw_positions: torch.Tensor | None,
    dims: torch.Tensor,
    degree: List[float],
    adjacency: List[Dict[int, float]],
    positions: List[Position | None],
    raw_context: Tuple[float, float, float, float] | None,
) -> List[dict]:
    assigned = set()
    items: List[dict] = []

    cluster_groups: Dict[int, List[int]] = {}
    for i in range(instance.block_count):
        if positions[i] is not None:
            continue
        gid = _group_id(instance, i, 3)
        if gid > 0:
            cluster_groups.setdefault(gid, []).append(i)

    for gid, members in cluster_groups.items():
        movable = [i for i in members if positions[i] is None]
        if len(movable) >= 2:
            item = _make_item(instance, movable, raw_positions, dims, degree, adjacency, raw_context)
            item["id"] = gid
            items.append(item)
            assigned.update(movable)

    for i in range(instance.block_count):
        if positions[i] is None and i not in assigned:
            item = _make_item(instance, [i], raw_positions, dims, degree, adjacency, raw_context)
            item["id"] = 100000 + i
            items.append(item)

    return items


def _make_item(
    instance: FloorSetInstance,
    blocks: List[int],
    raw_positions: torch.Tensor | None,
    dims: torch.Tensor,
    degree: List[float],
    adjacency: List[Dict[int, float]],
    raw_context: Tuple[float, float, float, float] | None,
) -> dict:
    ordered = _order_blocks(blocks, dims, degree, adjacency)
    rel = _raw_relative_rects(ordered, raw_positions, dims)
    if len(ordered) > 1 and _has_internal_overlap(rel):
        rel = _shelf_relative_rects(ordered, dims)

    raw_anchor = _raw_anchor(ordered, raw_positions)
    codes = [_boundary_code(instance, i) for i in ordered]
    phase = 0 if any(code & (1 | 8) for code in codes) else 2 if any(code & (2 | 4) for code in codes) else 1

    def build(rel_rects: List[Position]) -> dict:
        width = max((x + w for x, _, w, _ in rel_rects), default=0.0)
        height = max((y + h for _, y, _, h in rel_rects), default=0.0)
        return {
            "blocks": ordered,
            "rel": rel_rects,
            "width": width,
            "height": height,
            "area": sum(float(dims[i, 0] * dims[i, 1]) for i in ordered),
            "degree": sum(degree[i] for i in ordered),
            "boundary": sum(1 for code in codes if code != 0),
            "raw_anchor": raw_anchor,
            "compact_anchor": _compact_raw_anchor(ordered, raw_positions, raw_context),
            "phase": phase,
        }

    item = build(rel)
    if len(ordered) >= 4 and item["boundary"] == 0:
        variants = []
        seen = set()
        for candidate_rel in [rel, _shelf_relative_rects(ordered, dims, 0.75), _shelf_relative_rects(ordered, dims, 1.0), _shelf_relative_rects(ordered, dims, 1.2)]:
            key = tuple((round(x, 6), round(y, 6), round(w, 6), round(h, 6)) for x, y, w, h in candidate_rel)
            if key in seen:
                continue
            seen.add(key)
            variants.append(build(candidate_rel))
        item["variants"] = variants
    return item


def _order_blocks(
    blocks: List[int],
    dims: torch.Tensor,
    degree: List[float],
    adjacency: List[Dict[int, float]],
) -> List[int]:
    remaining = set(blocks)
    if not remaining:
        return []

    first = min(remaining, key=lambda i: (-float(dims[i, 0] * dims[i, 1]), -degree[i], i))
    ordered = [first]
    remaining.remove(first)

    while remaining:
        nxt = min(
            remaining,
            key=lambda i: (
                -sum(adjacency[i].get(j, 0.0) for j in ordered),
                -float(dims[i, 0] * dims[i, 1]),
                -degree[i],
                i,
            ),
        )
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def _raw_relative_rects(
    ordered: List[int],
    raw_positions: torch.Tensor | None,
    dims: torch.Tensor,
) -> List[Position]:
    raw_xy = []
    for i in ordered:
        raw = _raw_rect(raw_positions, i)
        if raw is None:
            raw_xy.append((0.0, 0.0))
        else:
            raw_xy.append((raw[0], raw[1]))

    min_x = min((x for x, _ in raw_xy), default=0.0)
    min_y = min((y for _, y in raw_xy), default=0.0)
    rel = []
    for i, (x, y) in zip(ordered, raw_xy):
        rel.append((x - min_x, y - min_y, float(dims[i, 0]), float(dims[i, 1])))
    return rel


def _shelf_relative_rects(ordered: List[int], dims: torch.Tensor, width_factor: float = 1.0) -> List[Position]:
    total_area = sum(float(dims[i, 0] * dims[i, 1]) for i in ordered)
    max_w = max((float(dims[i, 0]) for i in ordered), default=0.0)
    target_w = max(max_w, math.sqrt(max(total_area, EPS)) * width_factor)
    rel = []
    cursor_x = cursor_y = row_h = 0.0

    for i in ordered:
        w, h = float(dims[i, 0]), float(dims[i, 1])
        if cursor_x > 0.0 and cursor_x + w > target_w:
            cursor_y += row_h
            cursor_x = 0.0
            row_h = 0.0
        rel.append((cursor_x, cursor_y, w, h))
        cursor_x += w
        row_h = max(row_h, h)

    return rel


def _has_internal_overlap(rects: List[Position]) -> bool:
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _overlaps(rects[i], rects[j]):
                return True
    return False


def _raw_anchor(blocks: List[int], raw_positions: torch.Tensor | None) -> Tuple[float, float]:
    anchors = []
    for i in blocks:
        raw = _raw_rect(raw_positions, i)
        if raw is not None:
            anchors.append((raw[0], raw[1]))
    if not anchors:
        return 0.0, 0.0
    return min(x for x, _ in anchors), min(y for _, y in anchors)


def _compact_raw_context(
    instance: FloorSetInstance,
    raw_positions: torch.Tensor | None,
    dims: torch.Tensor,
) -> Tuple[float, float, float, float] | None:
    if raw_positions is None or raw_positions.numel() == 0:
        return None

    all_raw = []
    for i in range(instance.block_count):
        raw = _raw_rect(raw_positions, i)
        if raw is not None:
            all_raw.append(raw)
    if not all_raw:
        return None

    raw_min_x = min(x for x, _, _, _ in all_raw)
    raw_min_y = min(y for _, y, _, _ in all_raw)
    raw_max_x = max(x + w for x, _, w, _ in all_raw)
    raw_max_y = max(y + h for _, y, _, h in all_raw)
    raw_w = max(raw_max_x - raw_min_x, EPS)
    raw_h = max(raw_max_y - raw_min_y, EPS)

    total_area = max(sum(float(dims[i, 0] * dims[i, 1]) for i in range(instance.block_count)), EPS)
    aspect = max(min(raw_w / raw_h, 2.5), 0.4)
    target_w = math.sqrt(total_area * aspect)
    target_h = math.sqrt(total_area / aspect)
    scale_x = target_w / raw_w
    scale_y = target_h / raw_h
    return raw_min_x, raw_min_y, scale_x, scale_y


def _compact_raw_anchor(
    blocks: List[int],
    raw_positions: torch.Tensor | None,
    raw_context: Tuple[float, float, float, float] | None,
) -> Tuple[float, float] | None:
    if raw_positions is None or raw_context is None:
        return None
    raw_min_x, raw_min_y, scale_x, scale_y = raw_context

    anchors = []
    for i in blocks:
        raw = _raw_rect(raw_positions, i)
        if raw is not None:
            anchors.append(((raw[0] - raw_min_x) * scale_x, (raw[1] - raw_min_y) * scale_y))
    if not anchors:
        return None
    return min(x for x, _ in anchors), min(y for _, y in anchors)


def _item_rects(item: dict, x: float, y: float) -> List[Position]:
    return [(x + rx, y + ry, w, h) for rx, ry, w, h in item["rel"]]


def _place_item(
    instance: FloorSetInstance,
    item: dict,
    raw_positions: torch.Tensor | None,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    pin_adjacency: List[List[Tuple[float, float, float]]],
    dims: torch.Tensor,
) -> None:
    best_rects = None
    best_score = float("inf")

    for variant in item.get("variants", [item]):
        xs, ys = _candidate_origins(instance, variant, raw_positions, positions, adjacency, pin_adjacency, dims)
        for x, y in _candidate_origin_pairs(variant, xs, ys, positions):
            rects = _item_rects(variant, x, y)
            if not _can_place_many(rects, positions):
                continue
            score = _placement_score(instance, variant, rects, raw_positions, positions, adjacency, pin_adjacency)
            if score < best_score:
                best_score = score
                best_rects = rects

    if best_rects is None:
        best_rects = _fallback_place_item(item, positions)

    for block, rect in zip(item["blocks"], best_rects):
        positions[block] = rect


def _candidate_origins(
    instance: FloorSetInstance,
    item: dict,
    raw_positions: torch.Tensor | None,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    pin_adjacency: List[List[Tuple[float, float, float]]],
    dims: torch.Tensor,
) -> Tuple[List[float], List[float]]:
    raw_x, raw_y = item["raw_anchor"]
    xs = {0.0, raw_x}
    ys = {0.0, raw_y}
    compact_anchor = item.get("compact_anchor")
    if compact_anchor is not None:
        compact_x, compact_y = compact_anchor
        xs.add(float(compact_x))
        ys.add(float(compact_y))
    placed = [p for p in positions if p is not None]

    if placed:
        min_x = min(x for x, _, _, _ in placed)
        min_y = min(y for _, y, _, _ in placed)
        max_x = max(x + w for x, _, w, _ in placed)
        max_y = max(y + h for _, y, _, h in placed)
        xs.update({min_x, max_x, max_x - item["width"]})
        ys.update({min_y, max_y, max_y - item["height"]})

    for other in placed:
        ox, oy, ow, oh = other
        xs.update({ox, ox + ow, ox - item["width"], ox + ow - item["width"]})
        ys.update({oy, oy + oh, oy - item["height"], oy + oh - item["height"]})

    for block, (rx, ry, w, h) in zip(item["blocks"], item["rel"]):
        code = _boundary_code(instance, block)
        if placed:
            min_x = min(x for x, _, _, _ in placed)
            min_y = min(y for _, y, _, _ in placed)
            max_x = max(x + w2 for x, _, w2, _ in placed)
            max_y = max(y + h2 for _, y, _, h2 in placed)
            if code & 1:
                xs.add(min_x - rx)
            if code & 2:
                xs.add(max_x - rx - w)
            if code & 8:
                ys.add(min_y - ry)
            if code & 4:
                ys.add(max_y - ry - h)

        desired = _desired_xy_from_neighbors(block, raw_positions, positions, adjacency, pin_adjacency, dims)
        if desired is not None:
            dx, dy = desired
            xs.add(dx - rx)
            ys.add(dy - ry)

    anchor_x = [raw_x, 0.0]
    anchor_y = [raw_y, 0.0]
    if compact_anchor is not None:
        anchor_x.insert(0, float(compact_anchor[0]))
        anchor_y.insert(0, float(compact_anchor[1]))
    min_x_allowed, min_y_allowed = _origin_lower_bounds(instance)
    xs = {x for x in xs if x >= min_x_allowed - EPS}
    ys = {y for y in ys if y >= min_y_allowed - EPS}
    if not xs:
        xs = {min_x_allowed}
    if not ys:
        ys = {min_y_allowed}
    return _nearby_values(xs, anchor_x, 28), _nearby_values(ys, anchor_y, 28)


def _origin_lower_bounds(instance: FloorSetInstance) -> Tuple[float, float]:
    min_x = 0.0
    min_y = 0.0
    for i in range(instance.block_count):
        if _is_preplaced(instance, i):
            min_x = min(min_x, float(instance.target_positions[i, 0]))
            min_y = min(min_y, float(instance.target_positions[i, 1]))
    return min_x, min_y


def _candidate_origin_pairs(
    item: dict,
    xs: List[float],
    ys: List[float],
    positions: List[Position | None],
) -> List[Tuple[float, float]]:
    raw_x, raw_y = item["raw_anchor"]
    pairs = set()

    for x in xs:
        pairs.add((x, _lowest_y_for_item(item, x, positions)))

    for x in xs[:14]:
        for y in ys[:14]:
            pairs.add((x, y))

    ordered = sorted(
        pairs,
        key=lambda pair: (
            abs(pair[0] - raw_x) + abs(pair[1] - raw_y),
            pair[0] + item["width"],
            pair[1] + item["height"],
        ),
    )
    return ordered[:240]


def _desired_xy_from_neighbors(
    block: int,
    raw_positions: torch.Tensor | None,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    pin_adjacency: List[List[Tuple[float, float, float]]],
    dims: torch.Tensor,
) -> Tuple[float, float] | None:
    samples = []
    bw, bh = float(dims[block, 0]), float(dims[block, 1])

    for other, weight in adjacency[block].items():
        rect = positions[other]
        if rect is None:
            raw = _raw_rect(raw_positions, other)
            if raw is None:
                continue
            rect = raw
        ox, oy, ow, oh = rect
        samples.append((ox + 0.5 * ow - 0.5 * bw, oy + 0.5 * oh - 0.5 * bh, weight))

    for px, py, weight in pin_adjacency[block]:
        samples.append((px - 0.5 * bw, py - 0.5 * bh, weight))

    if not samples:
        return None
    total = sum(weight for _, _, weight in samples)
    if total <= 0.0:
        return None
    return (
        sum(x * weight for x, _, weight in samples) / total,
        sum(y * weight for _, y, weight in samples) / total,
    )


def _nearby_values(values, anchors: List[float], limit: int) -> List[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return [0.0]
    finite.sort(key=lambda value: min(abs(value - anchor) for anchor in anchors))
    return finite[:limit]


def _placement_score(
    instance: FloorSetInstance,
    item: dict,
    rects: List[Position],
    raw_positions: torch.Tensor | None,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    pin_adjacency: List[List[Tuple[float, float, float]]],
) -> float:
    all_rects = [p for p in positions if p is not None] + rects
    min_x = min(x for x, _, _, _ in all_rects)
    min_y = min(y for _, y, _, _ in all_rects)
    max_x = max(x + w for x, _, w, _ in all_rects)
    max_y = max(y + h for _, y, _, h in all_rects)
    bbox = (max_x - min_x) * (max_y - min_y)

    raw_dist = 0.0
    wire = 0.0
    boundary = 0.0
    item_pos = {block: rect for block, rect in zip(item["blocks"], rects)}

    for block, rect in item_pos.items():
        x, y, w, h = rect
        raw = _raw_rect(raw_positions, block)
        if raw is not None:
            raw_dist += abs(x - raw[0]) + abs(y - raw[1])

        code = _boundary_code(instance, block)
        if code & 1:
            boundary += 200.0 * abs(x - min_x)
        if code & 8:
            boundary += 200.0 * abs(y - min_y)
        if code & 2:
            boundary += 80.0 * abs((x + w) - max_x)
        if code & 4:
            boundary += 80.0 * abs((y + h) - max_y)

        cx, cy = x + 0.5 * w, y + 0.5 * h
        for other, weight in adjacency[block].items():
            other_rect = item_pos.get(other) or positions[other]
            if other_rect is None:
                continue
            ox, oy, ow, oh = other_rect
            wire += weight * (abs(cx - (ox + 0.5 * ow)) + abs(cy - (oy + 0.5 * oh)))

        for px, py, weight in pin_adjacency[block]:
            wire += weight * (abs(cx - px) + abs(cy - py))

    return bbox + 0.25 * wire + 0.01 * raw_dist + boundary


def _fallback_place_item(item: dict, positions: List[Position | None]) -> List[Position]:
    x_candidates = {0.0}
    for rect in positions:
        if rect is None:
            continue
        x, _, w, _ = rect
        x_candidates.add(x)
        x_candidates.add(x + w)

    best = None
    best_score = float("inf")
    for x in sorted(x_candidates):
        y = _lowest_y_for_item(item, x, positions)
        rects = _item_rects(item, x, y)
        if not _can_place_many(rects, positions):
            continue
        score = (x + item["width"]) * (y + item["height"])
        if score < best_score:
            best = rects
            best_score = score

    if best is not None:
        return best

    top = max((y + h for rect in positions if rect is not None for _, y, _, h in [rect]), default=0.0)
    return _item_rects(item, 0.0, top)


def _lowest_y_for_item(item: dict, x: float, positions: List[Position | None]) -> float:
    y = 0.0
    while True:
        bumped = False
        for probe_idx, probe in enumerate(_item_rects(item, x, y)):
            rel_y = item["rel"][probe_idx][1]
            for other in positions:
                if other is not None and _overlaps(probe, other):
                    _, oy, _, oh = other
                    y = max(y, oy + oh - rel_y)
                    bumped = True
                    break
            if bumped:
                break
        if not bumped:
            return y


def _single_block_polish(
    instance: FloorSetInstance,
    raw_positions: torch.Tensor | None,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    pin_adjacency: List[List[Tuple[float, float, float]]],
    dims: torch.Tensor,
    limit: int,
) -> None:
    movable = [
        i for i in range(instance.block_count)
        if positions[i] is not None and not _is_preplaced(instance, i) and _group_id(instance, i, 3) == 0
    ]
    movable.sort(
        key=lambda i: (
            _boundary_code(instance, i) == 0,
            -sum(adjacency[i].values()),
            -float(dims[i, 0] * dims[i, 1]),
            i,
        )
    )

    for block in movable[:limit]:
        old = positions[block]
        if old is None:
            continue

        item = {
            "blocks": [block],
            "rel": [(0.0, 0.0, old[2], old[3])],
            "width": old[2],
            "height": old[3],
            "raw_anchor": _raw_anchor([block], raw_positions),
        }
        xs, ys = _candidate_origins(instance, item, raw_positions, positions, adjacency, pin_adjacency, dims)
        positions[block] = None
        best = old
        best_score = _layout_score(instance, positions, adjacency, pin_adjacency, candidate=(block, old))
        shortlist = []

        for x, y in _candidate_origin_pairs(item, xs, ys, positions):
            candidate = (x, y, old[2], old[3])
            if not _can_place_many([candidate], positions):
                continue
            local_score = _placement_score(instance, item, [candidate], raw_positions, positions, adjacency, pin_adjacency)
            shortlist.append((local_score, candidate))

        shortlist.sort(key=lambda row: row[0])
        for _, candidate in shortlist[:10]:
            score = _layout_score(instance, positions, adjacency, pin_adjacency, candidate=(block, candidate))
            if score + 1e-9 < best_score:
                best = candidate
                best_score = score

        positions[block] = best


def _cluster_group_polish(
    instance: FloorSetInstance,
    raw_positions: torch.Tensor | None,
    raw_context: Tuple[float, float, float, float] | None,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    pin_adjacency: List[List[Tuple[float, float, float]]],
    dims: torch.Tensor,
    limit: int,
) -> None:
    groups: Dict[int, List[int]] = {}
    for i in range(instance.block_count):
        gid = _group_id(instance, i, 3)
        if gid > 0 and positions[i] is not None and not _is_preplaced(instance, i):
            groups.setdefault(gid, []).append(i)

    items = []
    for gid, members in groups.items():
        if len(members) < 2:
            continue
        item = _item_from_current_group(instance, members, raw_positions, raw_context, positions, adjacency, dims)
        item["id"] = gid
        items.append(item)

    items.sort(
        key=lambda item: (
            item["phase"],
            -item["boundary"],
            -item["degree"],
            -item["area"],
            item["id"],
        )
    )

    for item in items[:limit]:
        old_rects = [positions[i] for i in item["blocks"]]
        if any(rect is None for rect in old_rects):
            continue

        base_score = _layout_score(instance, positions, adjacency, pin_adjacency)
        for block in item["blocks"]:
            positions[block] = None

        xs, ys = _candidate_origins(instance, item, raw_positions, positions, adjacency, pin_adjacency, dims)
        shortlist = []
        for x, y in _candidate_origin_pairs(item, xs, ys, positions):
            rects = _item_rects(item, x, y)
            if not _can_place_many(rects, positions):
                continue
            local_score = _placement_score(instance, item, rects, raw_positions, positions, adjacency, pin_adjacency)
            shortlist.append((local_score, rects))

        best_rects = old_rects
        best_score = base_score
        shortlist.sort(key=lambda row: row[0])

        for _, rects in shortlist[:10]:
            for block, rect in zip(item["blocks"], rects):
                positions[block] = rect
            score = _layout_score(instance, positions, adjacency, pin_adjacency)
            for block in item["blocks"]:
                positions[block] = None
            if score + 1e-9 < best_score:
                best_score = score
                best_rects = rects

        for block, rect in zip(item["blocks"], best_rects):
            positions[block] = rect


def _item_from_current_group(
    instance: FloorSetInstance,
    members: List[int],
    raw_positions: torch.Tensor | None,
    raw_context: Tuple[float, float, float, float] | None,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    dims: torch.Tensor,
) -> dict:
    ordered = _order_blocks(members, dims, [sum(n.values()) for n in adjacency], adjacency)
    placed = [positions[i] for i in ordered]
    min_x = min(float(rect[0]) for rect in placed if rect is not None)
    min_y = min(float(rect[1]) for rect in placed if rect is not None)
    rel = []
    for block, rect in zip(ordered, placed):
        if rect is None:
            continue
        x, y, w, h = rect
        rel.append((x - min_x, y - min_y, w, h))

    width = max((x + w for x, _, w, _ in rel), default=0.0)
    height = max((y + h for _, y, _, h in rel), default=0.0)
    codes = [_boundary_code(instance, i) for i in ordered]
    phase = 0 if any(code & (1 | 8) for code in codes) else 2 if any(code & (2 | 4) for code in codes) else 1
    return {
        "blocks": ordered,
        "rel": rel,
        "width": width,
        "height": height,
        "area": sum(float(dims[i, 0] * dims[i, 1]) for i in ordered),
        "degree": sum(sum(adjacency[i].values()) for i in ordered),
        "boundary": sum(1 for code in codes if code != 0),
        "raw_anchor": _raw_anchor(ordered, raw_positions),
        "compact_anchor": _compact_raw_anchor(ordered, raw_positions, raw_context),
        "phase": phase,
    }


def _layout_score(
    instance: FloorSetInstance,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    pin_adjacency: List[List[Tuple[float, float, float]]],
    candidate: Tuple[int, Position] | None = None,
) -> float:
    rects = list(positions)
    if candidate is not None:
        block, rect = candidate
        rects[block] = rect

    placed = [p for p in rects if p is not None]
    if not placed:
        return 0.0

    min_x = min(x for x, _, _, _ in placed)
    min_y = min(y for _, y, _, _ in placed)
    max_x = max(x + w for x, _, w, _ in placed)
    max_y = max(y + h for _, y, _, h in placed)
    area = max((max_x - min_x) * (max_y - min_y), EPS)

    wire = 0.0
    for a, neighbors in enumerate(adjacency):
        pa = rects[a]
        if pa is None:
            continue
        ax, ay, aw, ah = pa
        acx, acy = ax + 0.5 * aw, ay + 0.5 * ah
        for b, weight in neighbors.items():
            if b <= a:
                continue
            pb = rects[b]
            if pb is None:
                continue
            bx, by, bw, bh = pb
            wire += weight * (abs(acx - (bx + 0.5 * bw)) + abs(acy - (by + 0.5 * bh)))

    for block, pins in enumerate(pin_adjacency):
        rect = rects[block]
        if rect is None:
            continue
        x, y, w, h = rect
        cx, cy = x + 0.5 * w, y + 0.5 * h
        for px, py, weight in pins:
            wire += weight * (abs(cx - px) + abs(cy - py))

    return area + 0.25 * wire + 100.0 * _boundary_violation_count(instance, rects)


def _boundary_violation_count(instance: FloorSetInstance, positions: List[Position | None]) -> int:
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
        code = _boundary_code(instance, i)
        if code == 0:
            continue
        x, y, w, h = rect
        missing_required_edge = (
            (code & 1 and abs(x - min_x) > 1e-6)
            or (code & 2 and abs(x + w - max_x) > 1e-6)
            or (code & 4 and abs(y + h - max_y) > 1e-6)
            or (code & 8 and abs(y - min_y) > 1e-6)
        )
        if missing_required_edge:
            count += 1
    return count


def _snap_boundary_blocks(
    instance: FloorSetInstance,
    positions: List[Position | None],
    adjacency: List[Dict[int, float]],
    pin_adjacency: List[List[Tuple[float, float, float]]],
) -> None:
    for _ in range(2):
        placed = [p for p in positions if p is not None]
        if not placed:
            return
        min_x = min(x for x, _, _, _ in placed)
        min_y = min(y for _, y, _, _ in placed)
        max_x = max(x + w for x, _, w, _ in placed)
        max_y = max(y + h for _, y, _, h in placed)

        for i, rect in enumerate(list(positions)):
            if rect is None or _is_preplaced(instance, i):
                continue
            if _group_id(instance, i, 3) > 0:
                continue
            code = _boundary_code(instance, i)
            if code == 0:
                continue
            x, y, w, h = rect
            nx, ny = x, y
            if code & 1:
                nx = min_x
            if code & 2:
                nx = max_x - w
            if code & 8:
                ny = min_y
            if code & 4:
                ny = max_y - h
            candidate = (nx, ny, w, h)
            base_violations = _boundary_violation_count(instance, positions)
            base_score = _layout_score(instance, positions, adjacency, pin_adjacency)
            positions[i] = None
            if _can_place_many([candidate], positions):
                positions[i] = candidate
                candidate_violations = _boundary_violation_count(instance, positions)
                candidate_score = _layout_score(instance, positions, adjacency, pin_adjacency)
                if not (candidate_violations < base_violations or candidate_score + 1e-9 < base_score):
                    positions[i] = rect
            else:
                positions[i] = rect


def _overlaps(a: Position, b: Position) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return min(ax + aw, bx + bw) - max(ax, bx) > EPS and min(ay + ah, by + bh) - max(ay, by) > EPS


def _can_place_many(candidate_rects: List[Position], positions: List[Position | None]) -> bool:
    for rect in candidate_rects:
        for other in positions:
            if other is not None and _overlaps(rect, other):
                return False
    for i in range(len(candidate_rects)):
        for j in range(i + 1, len(candidate_rects)):
            if _overlaps(candidate_rects[i], candidate_rects[j]):
                return False
    return True


def _assert_hard_legal(instance: FloorSetInstance, positions: List[Position]) -> None:
    if len(positions) != instance.block_count:
        raise ValueError(f"legalizer returned {len(positions)} positions for {instance.block_count} blocks")

    for i, rect in enumerate(positions):
        x, y, w, h = rect
        if not all(math.isfinite(v) for v in rect) or w <= 0.0 or h <= 0.0:
            raise ValueError(f"illegal rectangle for block {i}: {rect}")

        if _is_preplaced(instance, i):
            target = tuple(float(v) for v in instance.target_positions[i])
            if any(abs(a - b) > 1e-4 for a, b in zip(rect, target)):
                raise ValueError(f"preplaced block {i} moved: {rect} != {target}")
        elif _is_fixed(instance, i):
            tw, th = float(instance.target_positions[i, 2]), float(instance.target_positions[i, 3])
            if abs(w - tw) > 1e-4 or abs(h - th) > 1e-4:
                raise ValueError(f"fixed block {i} changed size: {(w, h)} != {(tw, th)}")
        else:
            area = max(float(instance.area_targets[i]), EPS)
            if abs(w * h - area) / area > 0.01:
                raise ValueError(f"soft block {i} violates area: {w * h} vs {area}")

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _overlaps(positions[i], positions[j]):
                raise ValueError(f"legalizer left overlap between {i} and {j}")
