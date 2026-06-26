"""Small PyTorch policy/value model for FloorSet grid placement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EdgeGNNPolicy(nn.Module):
    """Edge-message-passing policy inspired by the AlphaChip papers."""

    def __init__(
        self,
        node_feature_dim: int,
        grid_size: int,
        grid_cols: int | None = None,
        grid_rows: int | None = None,
        hidden_dim: int = 128,
        message_layers: int = 2,
    ):
        super().__init__()
        self.grid_size = grid_size
        if grid_cols is None or grid_rows is None:
            side = int(math.sqrt(grid_size))
            if side * side == grid_size:
                grid_cols = side
                grid_rows = side
            else:
                grid_cols = grid_size
                grid_rows = 1
        self.grid_cols = int(grid_cols)
        self.grid_rows = int(grid_rows)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.message_layers = int(message_layers)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("cell_features", self._build_cell_features(), persistent=False)

    def _build_cell_features(self) -> torch.Tensor:
        rows = torch.arange(self.grid_rows, dtype=torch.float32)
        cols = torch.arange(self.grid_cols, dtype=torch.float32)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        x = xx.reshape(-1) / max(self.grid_cols - 1, 1)
        y = yy.reshape(-1) / max(self.grid_rows - 1, 1)
        return torch.stack([x, y, x - 0.5, y - 0.5], dim=1)

    def _message_pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        for _ in range(self.message_layers):
            w = edge_weight.to(h.device).unsqueeze(1)
            msg_input = torch.cat([h[src], h[dst], w], dim=1)
            messages = self.edge_mlp(msg_input)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst.to(h.device), messages)
            deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            deg.index_add_(0, dst.to(h.device), torch.ones((messages.shape[0], 1), device=h.device))
            agg = agg / deg.clamp_min(1.0)
            h = self.node_norm(h + self.node_update(torch.cat([h, agg], dim=1)))
        return h

    def forward(self, observation: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = observation["node_features"].float()
        edge_index = observation["edge_index"].long()
        edge_weight = observation["edge_weight"].float()
        current_block = int(observation["current_block"].item())
        placed_mask = observation["placed_mask"].float()
        action_mask = observation.get("action_mask")

        h = self.node_encoder(node_features)
        h = self._message_pass(h, edge_index, edge_weight)
        graph_embed = h.mean(dim=0)
        if 0 <= current_block < h.shape[0]:
            current_embed = h[current_block]
        else:
            current_embed = torch.zeros_like(graph_embed)
        placed_fraction = placed_mask.mean().view(1).to(h.device)

        policy_context = torch.cat([graph_embed, current_embed, graph_embed - current_embed, placed_fraction])
        cells = self.cell_features.to(h.device)
        if cells.shape[0] != self.grid_size:
            cells = cells[: self.grid_size]
        policy_input = torch.cat(
            [
                policy_context.unsqueeze(0).expand(cells.shape[0], -1),
                cells,
            ],
            dim=1,
        )
        logits = self.policy_head(policy_input).squeeze(-1)
        if action_mask is not None:
            mask = action_mask.to(dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, -1e9)

        value_input = torch.cat([graph_embed, current_embed, placed_fraction])
        value = self.value_head(value_input).squeeze(-1)
        return logits, value


class CoordGNNPolicy(nn.Module):
    """Sequential GNN policy that predicts continuous normalized (x, y)."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        message_layers: int = 2,
        use_edge_gate: bool = False,
    ):
        super().__init__()
        self.use_edge_gate = bool(use_edge_gate)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_gate = None
        if self.use_edge_gate:
            self.edge_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, max(hidden_dim // 2, 1)),
                nn.ReLU(),
                nn.Linear(max(hidden_dim // 2, 1), 1),
                nn.Sigmoid(),
            )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.message_layers = int(message_layers)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _message_pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        for _ in range(self.message_layers):
            w = edge_weight.to(h.device).unsqueeze(1)
            msg_input = torch.cat([h[src], h[dst], w], dim=1)
            messages = self.edge_mlp(msg_input)
            agg = torch.zeros_like(h)
            dst_device = dst.to(h.device)
            if self.edge_gate is not None:
                gates = self.edge_gate(msg_input)
                messages = messages * gates
                agg.index_add_(0, dst_device, messages)
                normalizer = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
                normalizer.index_add_(0, dst_device, gates)
            else:
                agg.index_add_(0, dst_device, messages)
                normalizer = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
                normalizer.index_add_(0, dst_device, torch.ones((messages.shape[0], 1), device=h.device))
            agg = agg / normalizer.clamp_min(1.0)
            h = self.node_norm(h + self.node_update(torch.cat([h, agg], dim=1)))
        return h

    def forward(self, observation: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = observation["node_features"].float()
        edge_index = observation["edge_index"].long()
        edge_weight = observation["edge_weight"].float()
        current_block = int(observation["current_block"].item())
        placed_mask = observation["placed_mask"].float()

        h = self.node_encoder(node_features)
        h = self._message_pass(h, edge_index, edge_weight)
        graph_embed = h.mean(dim=0)
        if 0 <= current_block < h.shape[0]:
            current_embed = h[current_block]
        else:
            current_embed = torch.zeros_like(graph_embed)
        placed_fraction = placed_mask.mean().view(1).to(h.device)

        context = torch.cat([graph_embed, current_embed, graph_embed - current_embed, placed_fraction])
        xy = torch.sigmoid(self.coord_head(context))
        value_input = torch.cat([graph_embed, current_embed, placed_fraction])
        value = self.value_head(value_input).squeeze(-1)
        return xy, value
