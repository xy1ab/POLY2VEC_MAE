import os
import sys
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _CURRENT_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    V2Dataset = importlib.import_module("downstream_unet.loaders.loader").V2Dataset
else:
    from .loaders.loader import V2Dataset

# ==========================================
# 📐 评价指标与自定义 Loss 函数
# ==========================================
def calculate_iou(pred, target, threshold=0.5):
    """计算 IoU (交并比)，这是衡量几何修复质量的黄金标准"""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def total_variation_loss(img):
    """TV Loss: 严厉惩罚孤立像素点和噪点，强制输出大块平滑区域"""
    pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]
    pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]
    return torch.mean(torch.abs(pixel_dif1)) + torch.mean(torch.abs(pixel_dif2))

def spectral_consistency_loss(pred, target):
    """Spectral Loss: 强制预测图在频域上与真实图一致，促成闭环"""
    fft_pred = torch.fft.rfft2(pred)
    fft_target = torch.fft.rfft2(target)
    return torch.mean(torch.abs(fft_pred - fft_target))

def main():
    # ==========================================
    # 🌟 核心开关与配置
    # ==========================================
    TEST_MODE = False  # 正式训练保持 False
    
    data_dir = './data/v2_dataset_test' if TEST_MODE else './data/v2_dataset_full'
    save_dir = './checkpoints_v2'
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = 10 if TEST_MODE else 100
    batch_size = 16 if TEST_MODE else 32
    learning_rate = 2e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀训练启动！测试模式: {TEST_MODE}")

    # ==========================================
    # 1. 严谨的数据加载与切分
    # ==========================================
    dataset = V2Dataset(data_dir)
    total_len = len(dataset)
    if total_len == 0:
        raise ValueError(f"❌ 找不到数据，请检查 {data_dir} 是否已生成！")
        
    train_len = int(0.9 * total_len)
    val_len = int(0.05 * total_len)
    test_len = total_len - train_len - val_len
    
    # 锁定随机种子 42，确保每次切分的测试题不变
    train_set, val_set, test_set = random_split(
        dataset,[train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 封存 V2 的测试集索引
    torch.save(test_set.indices, './data/v2_test_indices.pt')
    print(f"📊 数据切分: 训练({train_len}) | 验证({val_len}) | 测试({test_len} 已封存)")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ==========================================
    # 2. 初始化 3 通道模型
    # ==========================================
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,         # [模糊图, 幅值图, 相位图]
        classes=1,
        activation='sigmoid'
    ).to(device)

    criterion_dice = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # ==========================================
    # 3. 循环训练
    # ==========================================
    best_iou = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- A. 训练阶段 ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
            
            # 计算三重 Loss
            l_dice = criterion_dice(pred, y)
            l_tv = total_variation_loss(pred)
            l_spec = spectral_consistency_loss(pred, y)
            
            # 加权组合
            loss = l_dice + 0.1 * l_tv + 0.05 * l_spec
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'TotLoss': f"{loss.item():.3f}", 'Dice': f"{l_dice.item():.3f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # --- B. 验证阶段 (严格计算 IoU) ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val  ]", leave=False):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                
                # 验证集也计算相同的 Loss 以便对比
                l_dice = criterion_dice(pred, y)
                l_tv = total_variation_loss(pred)
                l_spec = spectral_consistency_loss(pred, y)
                loss = l_dice + 0.1 * l_tv + 0.05 * l_spec
                
                val_loss += loss.item()
                val_iou += calculate_iou(pred, y)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # --- C. 成绩播报与存档 ---
        epoch_time = time.time() - start_time
        print(f"✨ Epoch {epoch+1} 结束 ({epoch_time:.1f}s)")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # 1. 保存最佳模型
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), os.path.join(save_dir, 'v2_unet_best.pth'))
            print(f"   🏆 发现最佳 V2 模型！当前最高 IoU: {best_iou:.4f}，已保存。")

        # 2. 每5轮保存一个定期备份
        if (epoch + 1) % 5 == 0:
            ckpt_name = f'v2_unet_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))
            print(f"   💾 定期备份已保存: {ckpt_name}")

    # 训练彻底结束后，保存最后一轮的快照
    torch.save(model.state_dict(), os.path.join(save_dir, 'v2_unet_last.pth'))
    print(f"🎉 训练全部完成！最优验证 IoU 锁定在: {best_iou:.4f}")

if __name__ == '__main__':
    main()
