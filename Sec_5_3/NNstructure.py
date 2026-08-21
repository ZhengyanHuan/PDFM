import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Embeds the flow-matching time t.

    Input:
        t: [B] or [B, 1]

    Output:
        emb: [B, dim]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        elif t.dim() != 1:
            raise ValueError(f"Expected t with shape [B] or [B,1], got {t.shape}")

        half = self.dim // 2
        device = t.device
        dtype = t.dtype

        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    """
    Projects the flow-time embedding into a conditioning vector.
    """
    def __init__(self, time_emb_dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


class FiLM(nn.Module):
    """
    Feature-wise affine modulation using the flow-time embedding.
    """
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1.0 + scale) + shift


class HybridSpaceTimeBlock(nn.Module):
    """
    Residual block for data shaped as [B, C, space, physical_time].

    Structure:
        1) spatial mixing with kernel (k, 1)
        2) temporal mixing with kernel (1, k)
        3) joint local space-time mixing with kernel (3, 3)
    """
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size_space: int = 5,
        kernel_size_time: int = 5,
        joint_kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        pad_s = kernel_size_space // 2
        pad_t = kernel_size_time // 2
        pad_joint = joint_kernel_size // 2
        groups = min(8, channels)

        self.norm1 = nn.GroupNorm(groups, channels)
        self.film1 = FiLM(cond_dim, channels)
        self.spatial_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size_space, 1),
            padding=(pad_s, 0),
        )

        self.norm2 = nn.GroupNorm(groups, channels)
        self.film2 = FiLM(cond_dim, channels)
        self.temporal_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size_time),
            padding=(0, pad_t),
        )

        self.norm3 = nn.GroupNorm(groups, channels)
        self.film3 = FiLM(cond_dim, channels)
        self.joint_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=joint_kernel_size,
            padding=pad_joint,
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x

        h = self.norm1(x)
        h = F.silu(h)
        h = self.film1(h, cond)
        h = self.spatial_conv(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.film2(h, cond)
        h = self.temporal_conv(h)

        h = self.norm3(h)
        h = F.silu(h)
        h = self.film3(h, cond)
        h = self.joint_conv(h)

        h = self.dropout(h)
        h = self.out_proj(h)

        return residual + h


class SpaceTimeModel(nn.Module):
    """
    Unified model with shared backbone and two modes:

    1) model_type="velocity_estimator"
       Input:
           x_t: [B, 1, H, W]
           t:   [B] or [B, 1]
       Output:
           v_t: [B, 1, H, W]

    2) model_type="classifier"
       Input:
           x_t: [B, 1, H, W]
           t:   [B] or [B, 1]
       Output:
           logits: [B]   (binary classification logit)

    Notes:
      - Single-channel input is fixed.
      - probabilities() applies sigmoid to binary logits.
      - For training the classifier, use BCEWithLogitsLoss.
    """
    def __init__(
        self,
        model_type: Literal["classifier", "velocity_estimator"],
        hidden_channels: int = 64,
        num_blocks: int = 5,
        time_emb_dim: int = 128,
        cond_dim: int = 256,
        kernel_size_space: int = 5,
        kernel_size_time: int = 5,
        joint_kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        if model_type not in {"classifier", "velocity_estimator"}:
            raise ValueError(
                f"model_type must be 'classifier' or 'velocity_estimator', got {model_type}"
            )

        self.model_type = model_type

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = TimeMLP(time_emb_dim, cond_dim)

        # fixed single-channel input
        self.input_proj = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([
            HybridSpaceTimeBlock(
                channels=hidden_channels,
                cond_dim=cond_dim,
                kernel_size_space=kernel_size_space,
                kernel_size_time=kernel_size_time,
                joint_kernel_size=joint_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)

        if self.model_type == "velocity_estimator":
            self.velocity_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        else:
            self.classifier_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier_head = nn.Linear(hidden_channels, 1)

    def backbone(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.dim() != 4 or x_t.shape[1] != 1:
            raise ValueError(f"x_t must have shape [B, 1, H, W], got {x_t.shape}")

        t_emb = self.time_embed(t)
        cond = self.time_mlp(t_emb)

        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, cond)

        h = self.final_norm(h)
        h = F.silu(h)
        return h

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x_t, t)

        if self.model_type == "velocity_estimator":
            v_t = self.velocity_head(h)  # [B, 1, H, W]
            return v_t

        logits = self.classifier_pool(h)             # [B, C, 1, 1]
        logits = torch.flatten(logits, start_dim=1)  # [B, C]
        logits = self.classifier_head(logits)        # [B, 1]
        logits = logits.squeeze(-1)                  # [B]
        return logits

    def predict_proba(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns probabilities for classifier mode.

        Output:
            probs: [B]
        """
        if self.model_type != "classifier":
            raise RuntimeError("predict_proba() is only valid when model_type='classifier'")

        logits = self.forward(x_t, t)   # [B]
        probs = torch.sigmoid(logits)   # [B]
        return probs







    