"""Encoder definitions for polygon MAE pretraining.

This module contains a clean ViT encoder that can be reused by downstream tasks
without exposing MAE-specific decoder logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .patch_embed import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed


class PolyEncoder(nn.Module):
    """Vision Transformer encoder for frequency-domain polygon images.

    Args:
        img_size: Input image size `(H, W)`.
        patch_size: Patch edge length.
        in_chans: Number of channels.
        embed_dim: Token embedding size.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (31, 31),
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 256,
        depth: int = 12,
        num_heads: int = 8,
    ):
        """Initialize encoder modules and positional parameters."""
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, 4.0, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize position embeddings and module weights."""
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=self.patch_embed.grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear/layernorm layers.

        Args:
            module: Layer module visited by `nn.Module.apply`.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor, ids_keep: torch.Tensor | None = None) -> torch.Tensor:
        """Encode patch tokens with optional MAE masking indices.

        Args:
            x: Input tensor `[B, C, H, W]`.
            ids_keep: Optional kept-token indices for MAE encoder path.

        Returns:
            Encoded sequence `[B, 1+L_keep, D]`.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full tokens without masking for downstream usage.

        Args:
            x: Input tensor `[B, C, H, W]`.

        Returns:
            Encoded sequence `[B, 1+L, D]`.
        """
        return self.forward_features(x, ids_keep=None)
