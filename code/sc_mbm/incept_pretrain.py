"""
Temporal-masking pre-training wrapper for InceptSADEncoder.

Instead of MAE-style patch masking, we mask random contiguous
time segments of the raw EEG input and train the encoder + a
lightweight decoder to reconstruct the full signal.
"""

import torch
import torch.nn as nn
import numpy as np


class InceptSADPretrain(nn.Module):
    """
    Pre-training wrapper.

    Forward returns (loss, pred, mask) to match MAEforEEG's interface
    so that the existing training loop in stageA1 can be reused with
    minimal changes.
    """

    def __init__(self, encoder, in_chans=64, time_len=512,
                 decoder_embed_dim=512, decoder_depth=4,
                 decoder_num_heads=8, mask_ratio=0.5):
        super().__init__()
        self.encoder = encoder
        self.in_chans = in_chans
        self.time_len = time_len
        self.mask_ratio = mask_ratio

        embed_dim = encoder.embed_dim
        seq_len = encoder.num_patches

        # ── lightweight decoder ──
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos = nn.Parameter(
            torch.zeros(1, seq_len, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # project back to input space: seq_len tokens → full time_len * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * (time_len // seq_len))

        self._init_pos_embed()

    def _init_pos_embed(self):
        seq_len = self.decoder_pos.shape[1]
        dim = self.decoder_pos.shape[2]
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.decoder_pos.data.copy_(pe.unsqueeze(0))

    # ── temporal masking ──────────────────────
    def _temporal_mask(self, x):
        """
        Masks random contiguous time segments of the raw EEG.
        x: [B, C, T]
        Returns: masked_x, mask  (mask: [B, T], 1 = masked, 0 = visible)
        """
        B, C, T = x.shape
        mask = torch.zeros(B, T, device=x.device)
        num_masked = int(T * self.mask_ratio)

        for i in range(B):
            # Choose a random start point for each sample
            # We use multiple smaller segments for diversity
            num_segments = max(1, num_masked // 32)
            seg_len = num_masked // num_segments
            for _ in range(num_segments):
                start = np.random.randint(0, max(1, T - seg_len))
                end = min(start + seg_len, T)
                mask[i, start:end] = 1.0

        masked_x = x.clone()
        masked_x = masked_x * (1 - mask.unsqueeze(1))  # zero out masked regions
        return masked_x, mask

    # ── decoder forward ──────────────────────
    def forward_decoder(self, latent):
        """latent: [B, seq_len, embed_dim] → [B, seq_len, in_chans * patch_t]"""
        x = self.decoder_embed(latent)
        x = x + self.decoder_pos
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x  # [B, seq_len, in_chans * patch_t]

    # ── loss ─────────────────────────────────
    def forward_loss(self, original, pred, mask):
        """
        MSE loss computed only on masked time regions.
        original: [B, C, T]
        pred: [B, seq_len, C * patch_t]  (needs reshaping)
        mask: [B, T]
        """
        B, C, T = original.shape
        seq_len = pred.shape[1]
        patch_t = T // seq_len

        # reshape prediction to [B, C, T]
        pred = pred.view(B, seq_len, C, patch_t)
        pred = pred.permute(0, 2, 1, 3).contiguous().view(B, C, T)

        # reshape mask to [B, 1, T] for broadcasting
        mask_expanded = mask.unsqueeze(1)  # [B, 1, T]

        # MSE only on masked regions
        diff = (pred - original) ** 2
        masked_diff = diff * mask_expanded
        num_masked = mask_expanded.sum() * C
        if num_masked == 0:
            return torch.tensor(0.0, device=original.device)
        loss = masked_diff.sum() / num_masked
        return loss

    # ── main forward ─────────────────────────
    def forward(self, x, img_features=None, valid_idx=None, mask_ratio=None):
        """
        Matches MAEforEEG.forward() signature:
        Returns: (loss, pred, mask)
        """
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio

        if x.ndim == 2:
            x = x.unsqueeze(0)

        # x: [B, C, T]
        masked_x, mask = self._temporal_mask(x)

        # encode
        latent = self.encoder(masked_x)  # [B, seq_len, embed_dim]

        # decode
        pred = self.forward_decoder(latent)  # [B, seq_len, C * patch_t]

        # loss
        loss = self.forward_loss(x, pred, mask)

        return loss, pred, mask


# ─────────────────────────────────────────────
# Decoder block (lightweight)
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop_p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=drop_p, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
