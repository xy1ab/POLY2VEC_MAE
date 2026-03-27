import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 解决路径依赖
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.loader_ocf import OCFDataset

def debug_sampling_and_range(data_path, num_samples=3):
    # 1. 初始化 Dataset (使用默认参数)
    dataset = OCFDataset(data_path, num_points=2048, boundary_ratio=0.7, jitter_std=0.01)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    
    for i in range(num_samples):
        # 随机取一个索引
        idx = np.random.randint(0, len(dataset))
        
        # 获取 Loader 输出的数据
        # v: [384], p: [N, 2], y: [N, 1]
        v, p, y = dataset[idx]
        
        p_np = p.numpy()
        y_np = y.numpy().flatten()
        
        # --- 诊断 A: 检查数值范围 ---
        p_min, p_max = p_np.min(axis=0), p_np.max(axis=0)
        print(f"样本 {idx} | 采样点 X范围: [{p_min[0]:.3f}, {p_max[0]:.3f}] | Y范围: [{p_min[1]:.3f}, {p_max[1]:.3f}]")
        
        # --- 诊断 B: 可视化采样点与标签 ---
        ax_scatter = axes[i, 0]
        # 内部点 (Label=1) 画绿色，外部点 (Label=0) 画红色
        mask_in = y_np > 0.5
        ax_scatter.scatter(p_np[~mask_in, 0], p_np[~mask_in, 1], c='red', s=1, alpha=0.3, label='Outside')
        ax_scatter.scatter(p_np[mask_in, 0], p_np[mask_in, 1], c='green', s=2, alpha=0.6, label='Inside')
        
        ax_scatter.set_title(f"Sample {idx}: Point Distribution & Labels")
        ax_scatter.set_xlim(-1.1, 1.1)
        ax_scatter.set_ylim(-1.1, 1.1)
        ax_scatter.grid(True, linestyle='--', alpha=0.5)
        ax_scatter.legend()

        # --- 诊断 C: 可视化三角形原始形态 (确认填充率) ---
        ax_gt = axes[i, 1]
        # 重新获取原始三角形数据进行绘制 (模拟 eval 逻辑)
        sample = dataset.data[idx]
        tris_raw = sample['triangles']
        meta = sample['meta']
        cx, cy, s = meta[0], meta[1], meta[2]
        # 关键验证：这里使用我们修改后的 s/2.0 公式
        tris_norm = (tris_raw - np.array([cx, cy])) / (s / 2.0 + 1e-9)
        
        for tri in tris_norm:
            poly = plt.Polygon(tri, facecolor='blue', edgecolor='black', alpha=0.4)
            ax_gt.add_patch(poly)
            
        ax_gt.set_title(f"Sample {idx}: GT Polygon (Norm: s/2)")
        ax_gt.set_xlim(-1, 1)
        ax_gt.set_ylim(-1, 1)
        ax_gt.grid(True)

    plt.tight_layout()
    save_path = "debug_loader_result.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ 诊断图已保存至: {save_path}")

if __name__ == "__main__":
    # 请根据你的实际路径修改
    DATA_PATH = "data/encoded_samples_20260324_1910.pt" 
    debug_sampling_and_range(DATA_PATH)