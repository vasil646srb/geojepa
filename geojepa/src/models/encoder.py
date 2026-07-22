"""Vision Transformer Encoder for GeoJEPA."""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, img_size=256, patch_size=16, in_channels=4, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//P, W//P)
        x = self.proj(x)
        # (B, embed_dim, num_patches_h, num_patches_w) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding with geo-aware initialization."""

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-based vision encoder for satellite imagery."""

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=4,
        embed_dim=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        self.pos_embed = PositionalEncoding(
            self.patch_embed.num_patches, embed_dim
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None, return_all_tokens=False):
        """
        Args:
            x: (B, C, H, W) input images
            mask: (B, num_patches) boolean mask, True = keep
            return_all_tokens: if True, return all patch tokens; else return mean pooled

        Returns:
            If return_all_tokens: (B, num_patches, embed_dim)
            Else: (B, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional encoding
        x = self.pos_embed(x)

        # Apply mask if provided (for JEPA context encoder)
        if mask is not None:
            # mask: True = keep, False = remove
            x = x * mask.unsqueeze(-1).float()

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_all_tokens:
            return x

        # Global average pooling
        return x.mean(dim=1)  # (B, embed_dim)


class EMAEncoder(nn.Module):
    """Exponential Moving Average of the target encoder."""

    def __init__(self, encoder, decay=0.996):
        super().__init__()
        self.encoder = encoder
        self.decay = decay

    @torch.no_grad()
    def update(self, student_encoder):
        """Update EMA parameters from student."""
        for ema_param, student_param in zip(
            self.encoder.parameters(), 
            student_encoder.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                student_param.data, alpha=1 - self.decay
            )

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
