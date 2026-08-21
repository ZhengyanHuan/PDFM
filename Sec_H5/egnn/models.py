import math
import torch
import torch.nn as nn
from egnn.egnn_new import EGNN, GNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np


# =========================
# CHANGED: new continuous time embedding for t in [0, 1]
# =========================
class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        # Accept scalar, [B], or [B,1]
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        if t.dim() == 0:
            t = t[None]
        elif t.dim() == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        elif t.dim() != 1:
            raise ValueError(f"Expected t scalar, [B], or [B,1], got {t.shape}")

        t = t.float()
        half = self.dim // 2
        if half == 0:
            return t[:, None]

        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half, device=t.device, dtype=t.dtype) /
            max(half - 1, 1)
        )
        args = (2.0 * math.pi) * t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100,
                 aggregation_method='sum', time_cond_dim=None):
        super().__init__()
        self.mode = mode
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time
        self.hidden_nf = hidden_nf
        self.time_cond_dim = (hidden_nf if time_cond_dim is None else time_cond_dim) if condition_time else 0

        if self.condition_time:
            self.time_embed = ContinuousTimeEmbedding(self.time_cond_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_cond_dim, self.time_cond_dim),
                nn.SiLU(),
                nn.Linear(self.time_cond_dim, self.time_cond_dim),
                nn.SiLU(),
            )
            self.time_to_input = nn.Linear(self.time_cond_dim, 1)

        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                time_cond_dim=self.time_cond_dim,
            )
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=3 + in_node_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method
            )

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims

        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]

        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, :self.n_dims].clone()

        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1, device=self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        time_cond_flat = None
        if self.condition_time:
            if np.prod(t.size()) == 1:
                t_batch = torch.full((bs,), t.item(), device=self.device, dtype=xh.dtype)
            else:
                t_batch = t.view(bs).to(device=self.device, dtype=xh.dtype)

            t_hidden = self.time_mlp(self.time_embed(t_batch))  # [bs, time_cond_dim]

            h_time = self.time_to_input(t_hidden)               # [bs, 1]
            h_time = h_time.view(bs, 1).repeat(1, n_nodes).view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

            time_cond_flat = t_hidden[:, None, :].repeat(1, n_nodes, 1)
            time_cond_flat = time_cond_flat.view(bs * n_nodes, self.time_cond_dim)

        if context is not None:
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(
                h, x, edges,
                node_mask=node_mask,
                edge_mask=edge_mask,
                time_cond=time_cond_flat,
            )
            vel = (x_final - x) * node_mask
            # vel = (x_final) * node_mask
            
            # print(x_final[0:29,0])
            # print(h_final[0:29,0])
            # vel = (x_final) * node_mask
            # print(vel[0:29,0])
            # print("---------")
        
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_stability_QM9(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        n_dims,
        hidden_nf=64,
        device='cpu',
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method='sum',
        pooling='mean',
        time_cond_dim=None,
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.pooling = pooling
        self.hidden_nf = hidden_nf
        self._edges_dict = {}

        self.in_node_nf = in_node_nf
        self.raw_node_nf = in_node_nf - int(condition_time)
        self.time_cond_dim = (hidden_nf if time_cond_dim is None else time_cond_dim) if condition_time else 0

        if self.condition_time:
            self.time_embed = ContinuousTimeEmbedding(self.time_cond_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_cond_dim, self.time_cond_dim),
                nn.SiLU(),
                nn.Linear(self.time_cond_dim, self.time_cond_dim),
                nn.SiLU(),
            )
            self.time_to_input = nn.Linear(self.time_cond_dim, 1)

        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            time_cond_dim=self.time_cond_dim,
        )

        graph_time_dim = self.time_cond_dim if condition_time else 1
        graph_in_dim = self.raw_node_nf + n_dims + graph_time_dim

        self.graph_head = nn.Sequential(
            nn.Linear(graph_in_dim, hidden_nf // 4),
            act_fn,
            nn.Linear(hidden_nf // 4, hidden_nf // 4),
            act_fn,
            nn.Linear(hidden_nf // 4, 1),
        )

        self.to(self.device)

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        return self._forward(t, xh, node_mask, edge_mask, context)

    def predict_proba(self, t, xh, node_mask, edge_mask, context=None):
        return torch.sigmoid(self._forward(t, xh, node_mask, edge_mask, context))

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims

        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]

        node_mask_flat = node_mask.view(bs * n_nodes, 1)
        edge_mask_flat = edge_mask.view(bs * n_nodes * n_nodes, 1)

        xh = xh.view(bs * n_nodes, -1).clone() * node_mask_flat
        x = xh[:, :self.n_dims].clone()
        h = xh[:, self.n_dims:].clone()

        if h_dims <= 0:
            raise ValueError("Expected node features in xh[..., 3:], but got none.")

        time_cond_flat = None
        if self.condition_time:
            if np.prod(t.size()) == 1:
                t_batch = torch.full((bs,), t.item(), device=self.device, dtype=xh.dtype)
            else:
                t_batch = t.view(bs).to(device=self.device, dtype=xh.dtype)

            t_hidden = self.time_mlp(self.time_embed(t_batch))  # [bs, time_cond_dim]

            h_time = self.time_to_input(t_hidden)               # [bs, 1]
            h_time = h_time.view(bs, 1).repeat(1, n_nodes).view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

            time_cond_flat = t_hidden[:, None, :].repeat(1, n_nodes, 1)
            time_cond_flat = time_cond_flat.view(bs * n_nodes, self.time_cond_dim)
        else:
            t_hidden = None

        if context is not None:
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        h_final, x_final = self.egnn(
            h, x, edges,
            node_mask=node_mask_flat,
            edge_mask=edge_mask_flat,
            time_cond=time_cond_flat,
        )

        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]
        if self.condition_time:
            h_final = h_final[:, :-1]

        h_final = h_final.view(bs, n_nodes, -1)
        x_final = x_final.view(bs, n_nodes, self.n_dims)
        node_mask_3d = node_mask.view(bs, n_nodes, 1)

        h_final = h_final * node_mask_3d
        x_final = x_final * node_mask_3d

        if self.pooling == 'sum':
            h_pool = h_final.sum(dim=1)
            x_pool = x_final.sum(dim=1)
        elif self.pooling == 'mean':
            denom = node_mask_3d.sum(dim=1).clamp(min=1.0)
            h_pool = h_final.sum(dim=1) / denom
            x_pool = x_final.sum(dim=1) / denom
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        if self.condition_time:
            t_graph = t_hidden.to(h_pool.dtype)      # [B, time_cond_dim]
        else:
            t_graph = t.view(bs, 1).to(h_pool.dtype) # [B, 1]

        graph_emb = torch.cat([h_pool, x_pool, t_graph], dim=1)
        logits = self.graph_head(graph_emb).squeeze(-1)
        return logits

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(device),
                    torch.LongTensor(cols).to(device)
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_stability_QM9_naive(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        n_dims,
        hidden_nf=64,
        device='cpu',
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method='sum',
        pooling='mean',
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.pooling = pooling

        # Naive per-node MLP input:
        # xh contains [x, h], optionally concatenate time and context.
        node_in_dim = n_dims + in_node_nf
        # if condition_time:
        #     node_in_dim += 1
        # if context_node_nf > 0:
        #     node_in_dim += context_node_nf
        # print(node_in_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf // 2),
            act_fn,
            nn.Linear(hidden_nf // 2, hidden_nf // 4),
            act_fn,
            nn.Linear(hidden_nf // 4, 1),
        )

        self.graph_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf//4),
            act_fn,
            nn.Linear(hidden_nf//4, hidden_nf//4),
            act_fn,
            nn.Linear(hidden_nf//4, 1),
        )


        self.to(self.device)

    def forward(self, t, xh, node_mask=None, edge_mask=None, context=None):
        return self._forward(t, xh, node_mask, edge_mask, context)

    def predict_proba(self, t, xh, node_mask=None, edge_mask=None, context=None):
        return torch.sigmoid(self._forward(t, xh, node_mask, edge_mask, context))

    def wrap_forward(self, node_mask=None, edge_mask=None, context=None):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    # def _forward(self, t, xh, node_mask, edge_mask, context):
    #     bs = xh.shape[0]
    
    #     dtype = next(self.graph_head.parameters()).dtype
    #     device = next(self.graph_head.parameters()).device
    
    #     graph_emb = torch.ones(
    #         bs,
    #         self.graph_head[0].in_features,
    #         device=device,
    #         dtype=dtype,
    #     )
    
    #     logits = self.graph_head(graph_emb).squeeze(-1)
    #     return logits

    def _forward(self, t, xh, node_mask=None, edge_mask=None, context=None):
        """
        Naive classifier.

        Inputs are kept compatible with the EGNN version:
        - t is used if condition_time=True
        - xh is used directly
        - node_mask is accepted but ignored
        - edge_mask is accepted but ignored
        - context is accepted and used only as extra raw features
        """
        bs, n_nodes, dims = xh.shape

        xh = xh.to(self.device)

        features = [xh]

        if self.condition_time:
            if torch.numel(t) == 1:
                t_node = torch.full(
                    (bs, n_nodes, 1),
                    float(t.item()),
                    device=xh.device,
                    dtype=xh.dtype,
                )
            else:
                t_node = t.view(bs, 1, 1).repeat(1, n_nodes, 1).to(
                    device=xh.device,
                    dtype=xh.dtype,
                )
            features.append(t_node)

        if context is not None:
            context = context.view(bs, n_nodes, self.context_node_nf).to(
                device=xh.device,
                dtype=xh.dtype,
            )
            features.append(context)

        node_input = torch.cat(features, dim=-1)       # [B, N, F]
        # print(node_input.shape)
        node_emb = self.node_mlp(node_input)           # [B, N, H]

        if self.pooling == 'sum':
            graph_emb = node_emb.sum(dim=1)            # [B, H]
        elif self.pooling == 'mean':
            graph_emb = node_emb.mean(dim=1)           # [B, H]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.graph_mlp(graph_emb).squeeze(-1) # [B]
        return logits





