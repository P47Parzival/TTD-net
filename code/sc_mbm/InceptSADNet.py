import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
import numpy as np


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


class Config(object):
    def __init__(self, dataset):
        self.model_name = 'InceptSADNet'
        self.data_path = dataset + '/raw'
        self.num_classes = 3
        self.f1_save_path = dataset + '/saved_dict/'  # 模型训练结果
        self.auc_save_path = dataset + '/saved_dict/'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt', encoding='utf-8').readlines()]  # 类别名单

        # for model
        self.learning_rate = 2e-3
        self.num_epoch = 100
        self.require_improvement = 1000
        self.batch_size = 128  # Increased from 32 for better GPU utilization
        self.dropout = 0.5

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block to recalibrate channel weights.
    Explicitly suppresses noisy channels.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvEmbedding(nn.Module):
    """
    Multi-Scale Temporal-Spatial extraction.
    Reverted to BatchNorm and SEBlock for Cross-Subject Stability.
    """
    def __init__(self, embed_size):
        super(ConvEmbedding, self).__init__()
        
        self.F1 = 8          
        self.D = 2           
        self.kernel_lengths = [15, 31, 63] 
        self.in_channels = 30 
        
        # 1. Multi-Scale Temporal Extraction
        self.branches = nn.ModuleList()
        for k in self.kernel_lengths:
            self.branches.append(nn.Sequential(
                nn.Conv2d(1, self.F1, (1, k), padding=(0, k//2), bias=False),
                nn.BatchNorm2d(self.F1), # Reverted to BatchNorm2d
                nn.ELU()
            ))
        
        # 2. Grouped Spatial Filtering
        total_filters = self.F1 * len(self.kernel_lengths)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                total_filters, 
                total_filters * self.D, 
                (self.in_channels, 1), 
                groups=total_filters, 
                bias=False
            ),
            nn.BatchNorm2d(total_filters * self.D), # Reverted to BatchNorm2d
            nn.ELU(),
            nn.AvgPool2d((1, 4)), 
            nn.Dropout(0.5) # Reverted to standard Dropout
        )

        # 3. SEBlock (Replaces CBAM)
        self.se_block = SEBlock(channels=total_filters * self.D)
        
        # 4. Final Projection
        self.projection = nn.Sequential(
            nn.Conv2d(total_filters * self.D, embed_size, (1, 1), bias=False),
            nn.BatchNorm2d(embed_size), # Reverted to BatchNorm2d
            nn.ELU(),
            nn.AvgPool2d((1, 4)), 
            nn.Dropout(0.5), # Reverted to standard Dropout
            Rearrange('b e (h) (w) -> b (h w) e'), 
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
            
        branches_out = [branch(x) for branch in self.branches]
        x = torch.cat(branches_out, dim=1) 
        
        x = self.spatial_conv(x)
        x = self.se_block(x) # Apply SEBlock
        
        out = self.projection(x)
        return out
                
class PatchEmbedding(nn.Sequential):
    def __init__(self, embed_size):
        super(PatchEmbedding, self).__init__()
        # [B, 1, C, S]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (30, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, embed_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.ConvNet(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Sequential):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res  # residual add
        return x


class MultiHeadAttention(nn.Sequential):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.embed_size = emb_size
        self.num_heads = num_heads
        self.K = nn.Linear(emb_size, emb_size)
        self.Q = nn.Linear(emb_size, emb_size)
        self.V = nn.Linear(emb_size, emb_size)
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.Q(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.K(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.V(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scale = self.embed_size ** (1 / 2)
        att = F.softmax(energy / scale, dim=-1)
        att = self.drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, embed_size, expansion, drop_p):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_size, expansion * embed_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * embed_size, embed_size)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embed_size, num_heads=5, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = ResidualAdd(nn.Sequential(
            nn.LayerNorm(embed_size),
            MultiHeadAttention(embed_size, num_heads, drop_p),
            nn.Dropout(drop_p),
        ))
        self.feedforward = ResidualAdd(nn.Sequential(
            nn.LayerNorm(embed_size),
            FeedForwardBlock(
                embed_size, expansion=forward_expansion, drop_p=forward_drop_p
            ),
            nn.Dropout(drop_p),
        ))

    def forward(self, x):
        x = self.attention(x)
        x = self.feedforward(x)
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, embed_size):
        super().__init__(*[TransformerEncoderBlock(embed_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, embed_size, n_classes):
        super(ClassificationHead, self).__init__()
        
        # Original SADNet global average pooling head (kept for compatibility)
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, n_classes),
        )
        
        # NEW GENTLE FUNNEL BOTTLENECK
        # 2480 -> LayerNorm -> 512 -> 64 -> 3
        self.fc = nn.Sequential(
            nn.LayerNorm(2480),   # <--- Stabilizes the massive flattened vector from the Transformer
            nn.Linear(2480, 512), # Gentler step down (was 256)
            nn.ELU(),
            nn.Dropout(0.5),      # Standard dropout
            nn.Linear(512, 64),   # Gentler step down (was 32)
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.shape[0], -1)
        out = self.fc(x)
        return out


class Model(nn.Sequential):
    def __init__(self, config, embed_size=40, depth=3, n_classes=3, **kwargs):
        super(Model, self).__init__()
        self.conv = ConvEmbedding(embed_size)
        # self.pos_embeding = Positional_Encoding(embed_size, 7, 0.5, config.device)
        self.transformer = TransformerEncoder(depth, embed_size)
        self.class_fc = ClassificationHead(embed_size, n_classes)
        self.config = config

    def forward(self, x):
        # [batch_size, 1, 30, 1001]
        x = x.permute(0, 2, 1)  # [batch_size channels sample]
        x = x.view(x.shape[0], 1, 30, -1)  # [batch_size, 1, channels, sample]
        x = self.conv(x)
        # print(x.shape)
        # x = self.pos_embeding(x)
        x = self.transformer(x)
        x = self.class_fc(x)
        return x


# 位置编码
class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        # out = x + nn.Parameter(self.pe, requires_grad=False)
        out = self.dropout(out)
        return out


class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
                self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
                self.__class__.__name__ +
                "(expression=%s) " % expression_str
        )