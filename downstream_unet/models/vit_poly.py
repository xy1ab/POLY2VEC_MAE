import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """生成 2D sine-cosine 位置编码"""
    grid_h, grid_w = grid_size
    grid_h_arr = np.arange(grid_h, dtype=np.float32)
    grid_w_arr = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_arr, grid_h_arr)
    grid = np.stack(grid, axis=0) # [2, H, W]
    
    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

import numpy as np

class MaskedAutoencoderViTPoly(nn.Module):
    """
    基于 ViT 的 Masked Autoencoder，针对频域双通道特征 (幅值、相位)
    默认规模为 ViT-Small: Encoder(384d, 12层) / Decoder(192d, 4层)
    """
    def __init__(self, img_size=(64, 32), patch_size=16, in_chans=2,
                 embed_dim=384, depth=12, num_heads=6,
                 decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 1. 编码器端
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 2. 解码器端
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # 重建预测通道，每个 patch 有 patch_size^2 * in_chans 个像素
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def random_masking(self, x, mask_ratio):
        """同时遮掩幅值和相位的Patch"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x shape: [B, 2, H, W]
        x = self.patch_embed(x) # [B, D, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2) # [B, L, D]
        
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        x = self.decoder_pred(x) # [B, L+1, p*p*C]
        x = x[:, 1:, :] # 去除 CLS Token
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        分别计算 幅值 和 相位 的重建 Loss
        imgs: [B, 2, H, W]
        """
        # Patchify 图像以匹配预测输出
        p = self.patch_size
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        c = imgs.shape[1] # 2
        
        target = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        target = torch.einsum('nchpwq->nhwcpq', target)
        target = target.reshape(shape=(imgs.shape[0], h * w, c * p**2)) # [B, L, p*p*2]
        
        # 将 target 分为两部分: 幅值 和 相位
        target_mag = target[:, :, :p**2]
        target_phase = target[:, :, p**2:]
        pred_mag = pred[:, :, :p**2]
        pred_phase = pred[:, :, p**2:]
        
        loss_mag = (pred_mag - target_mag) ** 2
        loss_mag = loss_mag.mean(dim=-1)
        loss_mag = (loss_mag * mask).sum() / mask.sum()
        
        loss_phase = (pred_phase - target_phase) ** 2
        loss_phase = loss_phase.mean(dim=-1)
        loss_phase = (loss_phase * mask).sum() / mask.sum()
        
        loss = loss_mag + loss_phase
        return loss, loss_mag, loss_phase

    def forward(self, mag, phase, mask_ratio=0.75):
        imgs = torch.cat([mag, phase], dim=1) # [B, 2, H, W]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss, loss_mag, loss_phase = self.forward_loss(imgs, pred, mask)
        return loss, loss_mag, loss_phase, pred, mask