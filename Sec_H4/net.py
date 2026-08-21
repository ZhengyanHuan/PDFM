import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        """
        t: [B] or [B, 1], usually in [0, 1]
        return: [B, dim]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class FingerprintVelocityNet(nn.Module):
    """
    Velocity network for flow matching on fingerprint images.

    Input:
        x_t: [B, 1, 96, 96]
        t:   [B] or [B, 1]

    Output:
        velocity: [B, 1, 96, 96]
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_dim=256):
        super().__init__()

        c = base_channels

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        self.enc1 = ResBlock(c, c, time_dim)
        self.down1 = nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1)

        self.enc2 = ResBlock(2 * c, 2 * c, time_dim)
        self.down2 = nn.Conv2d(2 * c, 4 * c, kernel_size=4, stride=2, padding=1)

        self.enc3 = ResBlock(4 * c, 4 * c, time_dim)
        self.down3 = nn.Conv2d(4 * c, 8 * c, kernel_size=4, stride=2, padding=1)

        self.enc4 = ResBlock(8 * c, 8 * c, time_dim)

        self.mid1 = ResBlock(8 * c, 8 * c, time_dim)
        self.mid2 = ResBlock(8 * c, 8 * c, time_dim)

        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(8 * c, 4 * c, time_dim)

        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(4 * c, 2 * c, time_dim)

        self.up1 = nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(2 * c, c, time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_t, t):
        t_emb = self.time_embed(t)

        h0 = self.init_conv(x_t)

        h1 = self.enc1(h0, t_emb)       # [B, c, 96, 96]
        h2 = self.enc2(self.down1(h1), t_emb)  # [B, 2c, 48, 48]
        h3 = self.enc3(self.down2(h2), t_emb)  # [B, 4c, 24, 24]
        h4 = self.enc4(self.down3(h3), t_emb)  # [B, 8c, 12, 12]

        h = self.mid1(h4, t_emb)
        h = self.mid2(h, t_emb)

        h = self.up3(h)
        h = self.dec3(torch.cat([h, h3], dim=1), t_emb)

        h = self.up2(h)
        h = self.dec2(torch.cat([h, h2], dim=1), t_emb)

        h = self.up1(h)
        h = self.dec1(torch.cat([h, h1], dim=1), t_emb)

        return self.out(h)


class FingerprintConstraintClassifier(nn.Module):
    """
    Classifier for estimating p(x_1 in C | x_t, t).

    Input:
        x_t: [B, 1, 96, 96]
        t:   [B] or [B, 1]

    Output:
        p:   [B], probability in [0, 1]
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        time_dim=256,
        # return_logits=False,
    ):
        super().__init__()

        # self.return_logits = return_logits
        c = base_channels

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        # Reduced by one level compared with the velocity U-Net.
        self.enc1 = ResBlock(c, c, time_dim)                # [B, c, 96, 96]
        self.down1 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)

        self.enc2 = ResBlock(2 * c, 2 * c, time_dim)        # [B, 2c, 48, 48]
        self.down2 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1)

        self.enc3 = ResBlock(4 * c, 4 * c, time_dim)        # [B, 4c, 24, 24]

        self.mid1 = ResBlock(4 * c, 4 * c, time_dim)
        self.mid2 = ResBlock(4 * c, 4 * c, time_dim)

        self.head = nn.Sequential(
            nn.GroupNorm(8, 4 * c),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * c, 2 * c),
            nn.SiLU(),
            nn.Linear(2 * c, 1),
        )

    def forward(self, x_t, t):
        t_emb = self.time_embed(t)

        h = self.init_conv(x_t)

        h = self.enc1(h, t_emb)
        h = self.down1(h)

        h = self.enc2(h, t_emb)
        h = self.down2(h)

        h = self.enc3(h, t_emb)
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        logits = self.head(h).squeeze(-1)  # [B]

        # if self.return_logits:
        return logits

    def predict_proba(self, x_t, t):
        logits = self.forward(x_t, t)
        return torch.sigmoid(logits)