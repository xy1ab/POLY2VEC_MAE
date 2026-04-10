# torchrun --nproc_per_node=4 train_0410.py --config configs/recons_unet.yaml
# python train_0410.py --config configs/recons_unet.yaml  # 单卡

import os
import sys
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
import argparse
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ==========================================
# 🛠️ 模块动态导入
# ==========================================
if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _CURRENT_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib
    loader_mod = importlib.import_module("downstream_unet.loaders.loader")
    ShardDataset = loader_mod.ShardDataset
    SingleDataset = loader_mod.SingleDataset
    # 确保你的 loader.py 中类名与此一致
    DistributedShardSampler = loader_mod.DistributedShardSampler 
else:
    from .loaders.loader import ShardDataset, SingleDataset, DistributedShardSampler


# ==========================================
# 📉 损失函数与指标
# ==========================================
def calculate_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def total_variation_loss(img):
    pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]
    pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]
    return torch.mean(torch.abs(pixel_dif1)) + torch.mean(torch.abs(pixel_dif2))

def spectral_consistency_loss(pred, target):
    fft_pred = torch.fft.rfft2(pred)
    fft_target = torch.fft.rfft2(target)
    return torch.mean(torch.abs(fft_pred - fft_target))

def build_arg_parser():
    parser = argparse.ArgumentParser(description="downstream_UNet")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser

# ==========================================
# 🚀 主训练逻辑
# ==========================================
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 读取配置
    data_dir = cfg['data']['data_dir']
    val_path = cfg['data']['val_path']
    batch_size = int(cfg['training']['batch_size'])
    epochs = int(cfg['training']['epochs'])
    learning_rate = float(cfg['training']['learning_rate'])
    save_dir = cfg['logging']['save_dir']
    save_freq = cfg['logging'].get('save_freq', 20)
    val_freq = cfg['training'].get('val_freq', 1)

    dice_weight = float(cfg['loss'].get('dice_weight', 1.0))
    tv_weight = float(cfg['loss'].get('tv_weight', 0.1))
    spec_weight = float(cfg['loss'].get('spec_weight', 0.05))

    # DDP 初始化
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1

    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"🚀 启动模式: {'DDP (' + str(world_size) + ' GPUs)' if is_ddp else '单卡'}")

    # ==========================================
    # 1. 数据加载 (Shard + Distributed)
    # ==========================================
    train_dataset = ShardDataset(data_dir)
    val_dataset = SingleDataset(val_path)

    # 使用自定义的分布式分片采样器
    train_sampler = DistributedShardSampler(
        train_dataset.shard_sizes,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )

    # 注意：batch_sampler 模式下不传 batch_size 和 shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,      # 强制 0，保护 32GB 内存
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ==========================================
    # 2. 模型与优化器
    # ==========================================
    model = smp.Unet(
        encoder_name=cfg['model'].get('encoder_name', 'resnet34'),
        encoder_weights=cfg['model'].get('encoder_weights', None),
        in_channels=cfg['model'].get('in_channels', 3),
        classes=cfg['model'].get('classes', 1),
        activation='sigmoid'
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion_dice = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # ==========================================
    # 3. 训练循环
    # ==========================================
    best_iou = 0.0
    start_time_total = time.time()

    for epoch in range(epochs):
        # 必须 set_epoch 以确保每轮洗牌顺序不同
        train_sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        else:
            pbar = train_loader

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            l_dice = criterion_dice(pred, y)
            l_tv = total_variation_loss(pred)
            l_spec = spectral_consistency_loss(pred, y)

            loss = (dice_weight * l_dice) + (tv_weight * l_tv) + (spec_weight * l_spec)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if local_rank == 0:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # --- 验证逻辑 ---
        val_loss, val_iou = 0.0, 0.0
        if (epoch + 1) % val_freq == 0:
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    
                    l_dice = criterion_dice(pred, y)
                    loss = dice_weight * l_dice + tv_weight * total_variation_loss(pred) + spec_weight * spectral_consistency_loss(pred, y)
                    
                    val_loss += loss.item()
                    val_iou += calculate_iou(pred, y)

        # --- DDP 指标同步 ---
        # 我们需要汇聚所有进程的 Loss 和 IoU
        stats = torch.tensor([train_loss, val_loss, val_iou], device=device)
        if is_ddp:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        # 计算全局平均
        # 每个进程的 train_loader 长度已经过 DistributedShardSampler 的对齐
        num_train_batches = len(train_loader) * world_size
        num_val_batches = len(val_loader) * world_size
        
        avg_train_loss = stats[0].item() / num_train_batches
        avg_val_loss = stats[1].item() / num_val_batches
        avg_val_iou = stats[2].item() / num_val_batches

        # --- 日志记录与模型保存 ---
        if local_rank == 0:
            epoch_time = time.time() - epoch_start
            print(f"✨ Epoch {epoch+1} | Time: {epoch_time:.1f}s | TrainLoss: {avg_train_loss:.4f} | ValIoU: {avg_val_iou:.4f}")

            state_dict = model.module.state_dict() if is_ddp else model.state_dict()
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                torch.save(state_dict, os.path.join(save_dir, 'unet_best.pth'))
                print(f"  🏆 New Best IoU: {best_iou:.4f}")

            if (epoch + 1) % save_freq == 0:
                torch.save(state_dict, os.path.join(save_dir, f'unet_epoch_{epoch+1}.pth'))

        # 每个 Epoch 结束后强制清理内存，这对 32GB 内存至关重要
        torch.cuda.empty_cache()

    if local_rank == 0:
        total_h = (time.time() - start_time_total) / 3600.0
        print(f"\n🎉 任务完成！Best IoU: {best_iou:.4f} | Total Time: {total_h:.2f}h")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()