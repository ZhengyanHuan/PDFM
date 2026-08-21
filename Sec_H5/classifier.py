import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B], expected in [0, 1]
        device = t.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return t.unsqueeze(-1)

        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class PairInteractionBlock(nn.Module):
    """
    One pairwise interaction layer.

    Inputs:
        h:          [B, N, H]
        x:          [B, N, 3]
        t_emb:      [B, Tdim]
        node_mask:  [B, N, 1]
        edge_mask:  [B, N, N, 1]
    Output:
        h_new:      [B, N, H]
    """
    def __init__(self, hidden_dim, time_dim, geom_dim=32, act_fn=nn.SiLU(), aggregation='mean'):
        super().__init__()
        self.aggregation = aggregation

        # geometry: rel(3) + dist2(1)
        self.geom_mlp = nn.Sequential(
            nn.Linear(4, geom_dim),
            act_fn,
            nn.Linear(geom_dim, geom_dim),
            act_fn,
            nn.Linear(geom_dim, geom_dim),
            act_fn,
            nn.Linear(geom_dim, geom_dim),
        )

        # time injection into edge / node updates
        self.time_to_edge = nn.Sequential(
            nn.Linear(time_dim, geom_dim),
            act_fn,
            nn.Linear(geom_dim, geom_dim),
            act_fn,
            nn.Linear(geom_dim, geom_dim)
        )

        self.time_to_node = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # edge message from h_i, h_j, geom_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + geom_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # node update from old h, aggregated message, time
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x, t_emb, node_mask, edge_mask):
        B, N, H = h.shape

        h_i = h.unsqueeze(2).expand(B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, H)

        x_i = x.unsqueeze(2)                                # [B, N, 1, 3]
        x_j = x.unsqueeze(1)                                # [B, 1, N, 3]
        rel = x_i - x_j                                     # [B, N, N, 3]
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)        # [B, N, N, 1]

        geom = self.geom_mlp(torch.cat([rel, dist2], dim=-1))   # [B, N, N, G]
        edge_time = self.time_to_edge(t_emb).unsqueeze(1).unsqueeze(1)  # [B,1,1,G]
        geom = geom + edge_time

        m_ij = self.edge_mlp(torch.cat([h_i, h_j, geom], dim=-1)) * edge_mask

        if self.aggregation == 'sum':
            m_i = m_ij.sum(dim=2)
        else:
            denom = edge_mask.sum(dim=2).clamp(min=1.0)
            m_i = m_ij.sum(dim=2) / denom

        node_time = self.time_to_node(t_emb).unsqueeze(1).expand(B, N, H)
        h_new = h + self.node_mlp(torch.cat([h, m_i, node_time], dim=-1))
        h_new = h_new * node_mask
        return h_new


class PairwiseStabilityClassifier(nn.Module):
    """
    Predict p(x_T is stable | x_t, t)

    Input:
        t:         [B]
        xh:        [B, N, D]   (for your case D=8 = 3 coords + 5 atom features)
        node_mask: [B, N, 1]
        edge_mask: [B, N, N, 1]
        context:   optional, currently unused

    Output:
        logits: [B]
        prob:   [B]
    """
    def __init__(
        self,
        in_node_nf=5,          # xh[..., 3:] dimension
        n_dims=3,
        hidden_dim=128,
        time_dim=32,
        geom_dim=64,
        n_interaction_layers=2,   # step 2 repeated twice
        aggregation='mean',
        pooling='mean',
        act_fn=nn.SiLU(),
    ):
        super().__init__()
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.pooling = pooling

        # diffusion-style time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            act_fn,
            nn.Linear(time_dim, time_dim),
        )

        # initial node encoding from x and h only
        self.node_encoder = nn.Sequential(
            nn.Linear(n_dims + in_node_nf, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # add time into node representation at input
        self.time_to_input = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            PairInteractionBlock(
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                geom_dim=geom_dim,
                act_fn=act_fn,
                aggregation=aggregation,
            )
            for _ in range(n_interaction_layers)
        ])

        # final graph head uses pooled node rep + geometry stats + time emb
        # geometry stats: mean, std, min pair distance
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim + 3 + time_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        return self._forward(t, xh, node_mask, edge_mask)

    def predict_proba(self, t, xh, node_mask, edge_mask, context=None):
        return torch.sigmoid(self._forward(t, xh, node_mask, edge_mask))

    def _forward(self, t, xh, node_mask, edge_mask):
        B, N, _ = xh.shape

        x = xh[:, :, :self.n_dims] * node_mask
        h_raw = xh[:, :, self.n_dims:] * node_mask

        # time embedding
        t_emb = self.time_embed(t)                               # [B, Tdim]

        # initial node encoding + time injection
        h = self.node_encoder(torch.cat([x, h_raw], dim=-1))    # [B, N, H]
        h = h + self.time_to_input(t_emb).unsqueeze(1)
        h = h * node_mask

        # remove self-edges from pair interactions
        eye = torch.eye(N, device=xh.device, dtype=edge_mask.dtype).view(1, N, N, 1)
        edge_mask_eff = edge_mask * (1.0 - eye)

        # repeated pairwise interaction blocks
        for block in self.blocks:
            h = block(h, x, t_emb, node_mask, edge_mask_eff)

        # masked node pooling
        if self.pooling == 'sum':
            h_pool = h.sum(dim=1)
        else:
            denom_nodes = node_mask.sum(dim=1).clamp(min=1.0)
            h_pool = h.sum(dim=1) / denom_nodes                 # [B, H]

        # masked geometry summary from valid pairs
        diff = x.unsqueeze(2) - x.unsqueeze(1)                  # [B, N, N, 3]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)      # [B, N, N]

        e_mask = edge_mask_eff.squeeze(-1)                      # [B, N, N]
        denom_edges = e_mask.sum(dim=(1, 2)).clamp(min=1.0)    # [B]

        dist_mean = (dist * e_mask).sum(dim=(1, 2)) / denom_edges

        dist_centered = (dist - dist_mean[:, None, None]) * e_mask
        dist_std = torch.sqrt((dist_centered ** 2).sum(dim=(1, 2)) / denom_edges + 1e-8)

        big = torch.full_like(dist, 1e9)
        dist_min = torch.where(e_mask > 0, dist, big).amin(dim=(1, 2))
        dist_min = torch.where(torch.isfinite(dist_min), dist_min, torch.zeros_like(dist_min))

        graph_feat = torch.cat(
            [
                h_pool,
                dist_mean.unsqueeze(-1),
                dist_std.unsqueeze(-1),
                dist_min.unsqueeze(-1),
                t_emb,
            ],
            dim=-1,
        )

        logits = self.graph_head(graph_feat).squeeze(-1)        # [B]
        return logits