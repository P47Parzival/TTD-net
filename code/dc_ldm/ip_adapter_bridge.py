"""
IP-Adapter style bridge for mapping EEG encoder output
to UNet cross-attention conditioning.

Instead of a simple nn.Linear, this uses:
1. A Resampler (Perceiver-style) to compress EEG tokens → fixed num_tokens
2. An MLP projection to match SDXL's context_dim (2048)

This replaces the old cond_stage_model's dim_mapper + channel_mapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Resampler(nn.Module):
    """
    Perceiver-style resampler: takes variable-length EEG token sequence
    and compresses it to a fixed number of output tokens via cross-attention.
    
    Input:  [B, seq_len, eeg_dim]    (e.g. [B, 32, 1024])
    Output: [B, num_tokens, out_dim]  (e.g. [B, 16, 2048])
    """
    def __init__(self, eeg_dim=1024, out_dim=2048, num_tokens=16,
                 num_heads=8, num_layers=2, ff_mult=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_tokens = nn.Parameter(torch.randn(1, num_tokens, out_dim) * 0.02)
        
        self.input_proj = nn.Linear(eeg_dim, out_dim) if eeg_dim != out_dim else nn.Identity()
        self.norm_in = nn.LayerNorm(out_dim)
        
        self.layers = nn.ModuleList([
            ResamplerLayer(out_dim, num_heads, ff_mult)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: [B, seq_len, eeg_dim]
        returns: [B, num_tokens, out_dim]
        """
        B = x.shape[0]
        x = self.input_proj(x)
        x = self.norm_in(x)
        
        latents = self.latent_tokens.expand(B, -1, -1)  # [B, num_tokens, out_dim]
        
        for layer in self.layers:
            latents = layer(latents, x)  # cross-attend: latents query, x is key/value
        
        return self.norm_out(latents)


class ResamplerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(0.1),
        )

    def forward(self, latents, context):
        """
        latents: [B, num_tokens, dim] (queries)
        context: [B, seq_len, dim]    (keys/values from EEG)
        """
        # Cross-attention
        normed_latents = self.norm1(latents)
        normed_context = self.norm_kv(context)
        attn_out, _ = self.cross_attn(normed_latents, normed_context, normed_context)
        latents = latents + attn_out
        
        # Feed-forward
        latents = latents + self.ff(self.norm2(latents))
        return latents


class IPAdapterBridge(nn.Module):
    """
    Full bridge: EEG encoder output → SDXL cross-attention conditioning.
    
    Pipeline:
        [B, seq_len, 1024] → Resampler → [B, num_tokens, 2048]
    
    Also computes CLIP alignment loss when image embeddings are provided.
    """
    def __init__(self, eeg_dim=1024, context_dim=2048, num_tokens=16,
                 clip_dim=768, use_clip_loss=True):
        super().__init__()
        self.context_dim = context_dim
        self.use_clip_loss = use_clip_loss
        
        # Main bridge: EEG → cross-attention tokens
        self.resampler = Resampler(
            eeg_dim=eeg_dim,
            out_dim=context_dim,
            num_tokens=num_tokens,
        )
        
        # CLIP alignment head (projects EEG to CLIP space for supervision)
        if use_clip_loss:
            self.clip_proj = nn.Sequential(
                nn.Linear(eeg_dim, eeg_dim),
                nn.GELU(),
                nn.Linear(eeg_dim, clip_dim),
            )
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
    
    def forward(self, eeg_latent):
        """
        eeg_latent: [B, seq_len, eeg_dim] from InceptSADEncoder
        returns: [B, num_tokens, context_dim] for UNet cross-attention
        """
        return self.resampler(eeg_latent)
    
    def get_clip_loss(self, eeg_latent, image_embeds):
        """
        Contrastive loss to align EEG embeddings with CLIP image embeddings.
        
        eeg_latent: [B, seq_len, eeg_dim]  (pooled to [B, eeg_dim])
        image_embeds: [B, clip_dim]
        """
        if not self.use_clip_loss:
            return torch.tensor(0.0, device=eeg_latent.device)
        
        # Pool EEG sequence to single vector
        eeg_pooled = eeg_latent.mean(dim=1)  # [B, eeg_dim]
        eeg_proj = self.clip_proj(eeg_pooled)  # [B, clip_dim]
        
        # Normalize
        eeg_proj = F.normalize(eeg_proj, dim=-1)
        image_embeds = F.normalize(image_embeds, dim=-1)
        
        # Contrastive loss (InfoNCE)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * eeg_proj @ image_embeds.t()  # [B, B]
        
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss
