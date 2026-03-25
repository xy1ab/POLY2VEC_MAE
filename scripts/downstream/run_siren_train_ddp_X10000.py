# export PYTHONPATH=$PYTHONPATH:.
# 单卡运行: python scripts/downstream/run_siren_train.py
# 多卡运行: torchrun --nproc_per_node=X scripts/downstream/run_siren_train.py

import os
import sys
import numpy
import yaml
import torch_musa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time

# ==========================================
# [新增 DDP 模块引入]
# ==========================================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 关键：将项目根目录加入搜索路径，确保能 import src 文件夹
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.downstream.task_recons.model_siren import FiLMSirenOCF
from src.downstream.task_recons.loader_ocf import OCFDataset

def calculate_iou(pred, target, threshold=0.5):
    """
    计算占用场点集的 Batch 平均 IoU
    pred: [B, N, 1], target: [B, N, 1]
    """
    p_bin = (pred > threshold).float()
    inter = (p_bin * target).sum(dim=1)
    union = p_bin.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def main():
    # ==========================================
    # 0. DDP 环境自动识别与初始化
    # ==========================================
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend='mccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.musa.set_device(local_rank)
        device = torch.device(f"musa:{local_rank}")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device('musa' if torch.musa.is_available() else 'cpu')
        world_size = 1

    # 定义主进程打印函数，防止多卡同时打印导致终端混乱
    def print_main(*args, **kwargs):
        if local_rank == 0:
            print(*args, **kwargs)

    # ==========================================
    # 1. 加载配置 (读取 configs/downstream/recons.yaml)
    # ==========================================
    config_path = os.path.join(ROOT_DIR, "configs/downstream/recons.yaml")
    if not os.path.exists(config_path):
        print_main(f"❌ 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    save_dir = cfg['save_dir']
    
    # 只有主进程负责创建文件夹
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    print_main(f"🚀 [OCF Training] 启动！模式: {'DDP 多卡' if is_ddp else '单卡'}")
    print_main(f"📍 实验结果将保存至: {os.path.abspath(save_dir)}")

    # ==========================================
    # 2. 数据准备 (30万级样本)
    # ==========================================
    dataset = OCFDataset(
        data_path=cfg['data_path'],
        num_points=cfg['num_points'],
        boundary_ratio=cfg['boundary_ratio']
    )
    
    total = len(dataset)
    torch.manual_seed(42) # 固定种子确保所有显卡的数据切分方式绝对一致
    
    # 按照 90/5/5 逻辑切分
    train_size = int(0.9 * total)
    val_size = int(0.05 * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # 只有主显卡负责封存测试集索引，防止文件读写冲突
    if local_rank == 0:
        torch.save(test_set.indices, cfg['test_indices_path'])
        print_main(f"📊 数据就绪: 训练({len(train_set)}) | 验证({len(val_set)}) | 测试({len(test_set)} 封存)")

    # --- DDP 数据分配 (核心改动) ---
    if is_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        # 注意：使用 sampler 时，DataLoader 的 shuffle 必须设为 False
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], sampler=val_sampler, num_workers=4)
    else:
        train_sampler = None
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    # ==========================================
    # 3. 初始化模型、优化器与损失函数
    # ==========================================
    model = FiLMSirenOCF(
        embed_dim=cfg['embed_dim'], 
        hidden_dim=cfg['hidden_dim'], 
        num_layers=cfg['num_layers'],
        omega_0=cfg['omega_0']
    ).to(device)

    # --- 给模型穿上 DDP 铠甲 ---
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCELoss()

    # ==========================================
    # 4. 核心训练循环
    # ==========================================
    best_val_iou = 0.0
    epochs = cfg['epochs']

    for epoch in range(epochs):
        if is_ddp:
            # 保证每轮数据打乱的随机性在各显卡间同步
            train_sampler.set_epoch(epoch)
            
        start_time = time.time()
        
        # --- A. 训练阶段 ---
        model.train()
        train_loss = 0.0
        
        # 只在主显卡上显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") if local_rank == 0 else train_loader
        
        for v, p, y in pbar:
            v, p, y = v.to(device), p.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(p, v)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if local_rank == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- B. 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val  ]", leave=False) if local_rank == 0 else val_loader
        
        with torch.no_grad():
            for v, p, y in val_pbar:
                v, p, y = v.to(device), p.to(device), y.to(device)
                preds = model(p, v)
                
                val_loss += criterion(preds, y).item()
                val_iou += calculate_iou(preds, y)
        
        # --- C. DDP 多卡数据同步汇总 ---
        # 每张卡算出自己的平均分
        avg_train_loss = torch.tensor(train_loss / len(train_loader), device=device)
        avg_val_loss = torch.tensor(val_loss / len(val_loader), device=device)
        avg_val_iou = torch.tensor(val_iou / len(val_loader), device=device)

        if is_ddp:
            # 将所有卡的分数加起来
            dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_val_iou, op=dist.ReduceOp.SUM)
            # 除以显卡数量，得到全局真实的平均分
            avg_train_loss /= world_size
            avg_val_loss /= world_size
            avg_val_iou /= world_size

        # 将 tensor 转换回 python float 数字
        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item()
        avg_val_iou = avg_val_iou.item()
        
        # 更新学习率调度器 (使用全局平均 Val Loss)
        scheduler.step(avg_val_loss)

        # --- D. 日志播报与存档 (仅主进程) ---
        if local_rank == 0:
            epoch_time = time.time() - start_time
            print_main(f"✨ Epoch {epoch+1} 结束 ({epoch_time:.1f}s)")
            print_main(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_iou:.4f}")

            # 提取真实的 model (脱下 DDP 的机甲外壳)
            model_to_save = model.module if is_ddp else model

            # 保存表现最好的模型
            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'siren_ocf_best.pth'))
                print_main(f"   🏆 发现更强模型！已保存至 siren_ocf_best.pth")

            # 每 10 轮保存一个定期快照
            if (epoch + 1) % 10 == 0:
                torch.save(model_to_save.state_dict(), os.path.join(save_dir, f'siren_ocf_epoch_{epoch+1}.pth'))

    print_main(f"\n🎉 训练全部完成！最高验证 mIoU: {best_val_iou:.4f}")

    # 训练结束，安全销毁通信进程组
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()