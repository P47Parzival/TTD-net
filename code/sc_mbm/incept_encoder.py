"""
InceptSADEncoder: A drop-in replacement for eeg_encoder using
InceptSADNet's multi-scale temporal extraction with SE attention.

Designed for 64-channel EEG input (PhysioNet / Things-EEG).
Exposes the same interface as eeg_encoder so that cond_stage_model
in ldm_for_eeg.py can consume it without changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# ─────────────────────────────────────────────
# Building blocks (adapted from InceptSADNet)
# ─────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: recalibrates channel weights."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvEmbedding64(nn.Module):
    """
    Multi-scale temporal-spatial extraction for 64-channel EEG.
    
    Input : [B, 1, 64, time_len]
    Output: [B, seq_len, embed_dim]
    """
    def __init__(self, embed_dim=1024, in_channels=64, time_len=512,
                 kernel_lengths=None, F1=8, D=2, dropout=0.5):
        super().__init__()
        if kernel_lengths is None:
            kernel_lengths = [7, 15, 31]

        self.F1 = F1
        self.D = D
        self.in_channels = in_channels

        # 1. Multi-Scale Temporal Extraction
        self.branches = nn.ModuleList()
        for k in kernel_lengths:
            self.branches.append(nn.Sequential(
                nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.ELU()
            ))

        # 2. Grouped Spatial Filtering  (collapses channel dim)
        total_filters = F1 * len(kernel_lengths)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                total_filters,
                total_filters * D,
                (in_channels, 1),       # (64, 1) — collapses spatial dim
                groups=total_filters,
                bias=False
            ),
            nn.BatchNorm2d(total_filters * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        # 3. SE block
        self.se_block = SEBlock(channels=total_filters * D)

        # 4. Final Projection → embed_dim
        self.projection = nn.Sequential(
            nn.Conv2d(total_filters * D, embed_dim, (1, 1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            Rearrange('b e h w -> b (h w) e'),   # → [B, seq_len, embed_dim]
        )

        # ── compute output sequence length for downstream use ──
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, time_len)
            dummy_out = self._forward_conv(dummy)
            self._seq_len = dummy_out.shape[1]

    def _forward_conv(self, x):
        branches_out = [branch(x) for branch in self.branches]
        x = torch.cat(branches_out, dim=1)
        x = self.spatial_conv(x)
        x = self.se_block(x)
        x = self.projection(x)
        return x

    @property
    def seq_len(self):
        return self._seq_len

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)         # [B, C, T] → [B, 1, C, T]
        return self._forward_conv(x)


# ─────────────────────────────────────────────
# Transformer blocks (from InceptSADNet)
# ─────────────────────────────────────────────

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.3):
        super().__init__()
        self.embed_size = emb_size
        self.num_heads = num_heads
        self.K = nn.Linear(emb_size, emb_size)
        self.Q = nn.Linear(emb_size, emb_size)
        self.V = nn.Linear(emb_size, emb_size)
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        queries = rearrange(self.Q(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(self.K(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.V(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scale = self.embed_size ** 0.5
        att = F.softmax(energy / scale, dim=-1)
        att = self.drop(att)

        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.projection(out)


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_size, expansion=4, drop_p=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_size, expansion * embed_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * embed_size, embed_size)
        )

    def forward(self, x):
        return self.fc(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads=8, drop_p=0.3,
                 forward_expansion=4, forward_drop_p=0.3):
        super().__init__()
        self.attention = ResidualAdd(nn.Sequential(
            nn.LayerNorm(embed_size),
            MultiHeadAttention(embed_size, num_heads, drop_p),
            nn.Dropout(drop_p),
        ))
        self.feedforward = ResidualAdd(nn.Sequential(
            nn.LayerNorm(embed_size),
            FeedForwardBlock(embed_size, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p),
        ))

    def forward(self, x):
        x = self.attention(x)
        x = self.feedforward(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, embed_size, num_heads=8, drop_p=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_size, num_heads, drop_p)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ─────────────────────────────────────────────
# Main encoder  (drop-in for eeg_encoder)
# ─────────────────────────────────────────────

class InceptSADEncoder(nn.Module):
    """
    Drop-in replacement for eeg_encoder.

    Interface contract:
        .num_patches  – int, sequence length produced by the encoder
        .embed_dim    – int, embedding dimension
        .forward(x)   – returns  [B, num_patches, embed_dim]  if global_pool=False
                                 [B, 1,           embed_dim]  if global_pool=True
        .load_checkpoint(state_dict) – loads pre-trained weights
    """

    def __init__(self, time_len=512, in_chans=64, embed_dim=1024,
                 depth=6, num_heads=8, global_pool=True,
                 kernel_lengths=None, F1=8, D=2, dropout=0.3,
                 # unused params kept for config compatibility
                 patch_size=4, mlp_ratio=1.0, **kwargs):
        super().__init__()

        self.conv_embed = ConvEmbedding64(
            embed_dim=embed_dim,
            in_channels=in_chans,
            time_len=time_len,
            kernel_lengths=kernel_lengths,
            F1=F1, D=D, dropout=dropout
        )

        self.transformer = TransformerEncoder(depth, embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.global_pool = global_pool

        # ── interface attributes consumed by cond_stage_model ──
        self.num_patches = self.conv_embed.seq_len
        self.embed_dim = embed_dim

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────
    def forward(self, x):
        """
        Args:
            x: [B, in_chans, time_len]  or  [B, time_len] (auto-unsqueezed)
        Returns:
            latent: [B, 1, embed_dim] if global_pool else [B, num_patches, embed_dim]
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x = self.conv_embed(x)            # [B, seq_len, embed_dim]
        x = self.transformer(x)           # [B, seq_len, embed_dim]

        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)  # [B, 1, embed_dim]

        x = self.norm(x)
        return x

    # ── checkpoint loading ───────────────────
    def load_checkpoint(self, state_dict):
        """Load pre-trained encoder weights (compatible with both
        InceptSADPretrain and standalone encoder checkpoints)."""
        # Filter out decoder keys if loading from a pretrain wrapper
        filtered = {k.replace('encoder.', ''): v
                    for k, v in state_dict.items()
                    if not k.startswith('decoder')}
        m, u = self.load_state_dict(filtered, strict=False)
        print('InceptSADEncoder – missing keys:', u)
        print('InceptSADEncoder – unexpected keys:', m)
