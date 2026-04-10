import os
import sys
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import argparse
import yaml
from pathlib import Path

# ==========================================
# 🛠️ 模块动态导入
# ==========================================
if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _CURRENT_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import importlib
    # 导入你最新的 SingleDataset
    SingleDataset = importlib.import_module("downstream_unet.loaders.loader").SingleDataset
else:
    from .loaders.loader import SingleDataset


def calculate_metrics(pred, target, input_blur, threshold=0.5):
    """计算硬核数据指标"""
    pred_bin = (pred > threshold).astype(np.float32)
    
    # 1. IoU
    intersection = np.sum(pred_bin * target)
    union = np.sum(pred_bin) + np.sum(target) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # 2. MSE 改善对比
    mse_before = np.mean((input_blur - target) ** 2)
    mse_after = np.mean((pred - target) ** 2)
    
    # 3. 绝对错判的像素个数
    wrong_pixels = np.sum(np.abs(pred_bin - target))
    
    # 4. 犹豫像素数 (0.1 到 0.9 之间的模糊灰色地带)
    uncertain_before = np.sum((input_blur > 0.1) & (input_blur < 0.9))
    uncertain_after = np.sum((pred > 0.1) & (pred < 0.9))
    
    return iou, mse_before, mse_after, wrong_pixels, uncertain_before, uncertain_after

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_time = time.strftime("%Y%m%d_%H%M%S")
    
    # ========== 1. 读取 YAML 配置 ==========
    config_path = './configs/recons_unet.yaml'
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 适配你最新的 YAML 字段
    val_data_path = cfg['data']['val_path']  # "./data/mask75/val.pt"
    model_dir = cfg['logging']['save_dir']   # "./unet_checkpoints/mask75"
    model_path = os.path.join(model_dir, 'unet_best.pth')
    
    # 可视化输出目录
    vis_dir = './vis_results_v2'
    os.makedirs(vis_dir, exist_ok=True)
    report_path = os.path.join(vis_dir, f'evaluation_report_{current_time}.txt')
    
    # ========== 2. 加载验证/测试集 ==========
    # 注意：现在直接加载整个验证集文件，不再使用 indices 过滤
    try:
        test_set = SingleDataset(val_data_path)
        print(f"\n📦 成功加载验证集: {val_data_path} | 样本数: {len(test_set)}")
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return
    
    # ========== 3. 模型初始化 ==========
    model = smp.Unet(
        encoder_name=cfg['model'].get('encoder_name', 'resnet34'),
        encoder_weights=None, # 推理时不需要预训练权重下载
        in_channels=cfg['model'].get('in_channels', 3),
        classes=cfg['model'].get('classes', 1),
        activation='sigmoid'
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"🤖 成功加载最佳模型权重: {model_path}")
    else:
        print(f"⚠️ 找不到权重文件 {model_path}，将使用随机初始化模型进行测试！")
    
    model.eval()
    print("="*50)
    
    # ========== 4. 随机抽取并可视化 ==========
    num_samples = 4 # 增加到 4 个样本，让报告更丰富
    sample_indices = random.sample(range(len(test_set)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    
    total_iou = 0.0
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"评估时间: {current_time}\n模型路径: {model_path}\n")
        f.write("-" * 50 + "\n")
        
        for i, idx in enumerate(sample_indices):
            x, y = test_set[idx]
            
            with torch.no_grad():
                # x shape: [3, 256, 256] -> [1, 3, 256, 256]
                pred = model(x.unsqueeze(0).to(device))
            
            # 数据解析（对应你的 3 通道：Mag/Cos/Sin）
            img_blur = x[0].numpy()       # 通道0：通常是作为输入的模糊图/空间图
            img_mag  = x[1].numpy()       # 通道1：幅值图
            img_pred = pred.squeeze().cpu().numpy()
            img_gt   = y.squeeze().numpy()
            
            iou, mse_bef, mse_aft, wrong_px, unc_bef, unc_aft = calculate_metrics(
                img_pred, img_gt, img_blur
            )
            total_iou += iou
            
            # 打印与记录
            res_msg = (f"Index {idx:04d} | IoU: {iou*100:5.2f}% | "
                       f"MSE: {mse_bef:.4f}->{mse_aft:.4f} | "
                       f"Wrong Px: {int(wrong_px)}")
            print(res_msg)
            f.write(res_msg + "\n")
            
            # 绘图逻辑
            axes[i, 0].imshow(img_blur, cmap='gray')
            axes[i, 0].set_title(f"Input (CH0)\nUnclear: {int(unc_bef)}")
            
            axes[i, 1].imshow(img_mag, cmap='viridis')
            axes[i, 1].set_title("Frequency Mag (CH1)")
            
            # 预测图：增加阈值二值化展示效果更好
            axes[i, 2].imshow(img_pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f"Prediction\nIoU: {iou*100:.2f}%")
            
            axes[i, 3].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title(f"Ground Truth\nError Px: {int(wrong_px)}")
            
            for ax in axes[i]: ax.axis('off')
            
        avg_iou = total_iou / num_samples
        f.write(f"\n平均 IoU: {avg_iou*100:.2f}%\n")
        print(f"\n🎯 抽测均值 IoU: {avg_iou*100:.2f}%")
    
    img_save_path = os.path.join(vis_dir, f'eval_vis_{current_time}.png')
    plt.savefig(img_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ 结果已保存至: {vis_dir}")

if __name__ == '__main__':
    main()