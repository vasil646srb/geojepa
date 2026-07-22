"""JEPA Predictor Network for GeoJEPA."""

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """
    Predictor network that predicts target embeddings from context embeddings.
    Uses cross-attention between context tokens and learnable target queries.
    """

    def __init__(
        self,
        embed_dim=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        num_patches=256
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Learnable mask tokens for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional encoding for predictor
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.predictor_pos_embed, std=0.02)

        # Predictor transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            PredictorBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Projection head to match target encoder dimensions
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, context_tokens, target_mask_indices):
        """
        Args:
            context_tokens: (B, N_context, embed_dim) from context encoder
            target_mask_indices: (B, N_target) indices of masked patches to predict

        Returns:
            predicted_embeddings: (B, N_target, embed_dim)
        """
        B = context_tokens.shape[0]
        N_target = target_mask_indices.shape[1]

        # Create target queries with mask tokens + positional encoding
        target_queries = self.mask_token.expand(B, N_target, -1)  # (B, N_target, embed_dim)

        # Add positional encoding for target positions
        pos_embed = self.predictor_pos_embed.expand(B, -1, -1)
        target_pos = torch.gather(
            pos_embed, 1, 
            target_mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )
        target_queries = target_queries + target_pos

        # Concatenate context and target for cross-attention
        # Context tokens attend to each other, target queries attend to context
        x = torch.cat([context_tokens, target_queries], dim=1)

        # Run through predictor blocks
        for block in self.blocks:
            x = block(x, len_context=context_tokens.shape[1])

        x = self.norm(x)

        # Extract target predictions
        target_pred = x[:, context_tokens.shape[1]:, :]  # (B, N_target, embed_dim)
        target_pred = self.proj(target_pred)

        return target_pred


class PredictorBlock(nn.Module):
    """Predictor block with causal masking for target positions."""

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

    def forward(self, x, len_context):
        """
        Args:
            x: concatenated [context_tokens, target_queries]
            len_context: number of context tokens
        """
        B, N, D = x.shape

        # Create causal mask: target positions can attend to context + previous targets
        # For simplicity, we allow full attention within context and cross-attention from targets to all
        attn_mask = None  # Full attention for now (can be made causal for autoregressive prediction)

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x
