import os
import sys
import torch
import numpy as np

# 1. 兼容性补丁：防止加载 Numpy 2.x 数据时在 1.x 环境报错
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core

def inspect_data(data_path):
    print(f"📦 正在载入数据: {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ 找不到文件: {data_path}，请检查路径。")
        return

    # 加载数据
    data = torch.load(data_path, weights_only=False, map_location='cpu')
    print(f"✅ 成功加载数据集，总样本数: {len(data)}\n")
    
    # 随机抽取第0个样本进行切片分析
    sample = data[0]
    print("=" * 40)
    print("🔍 [样本 0] 数据结构概览:")
    print("包含的键值:", list(sample.keys()))
    
    # ---------------------------------------------------------
    # A. 检查 Embedding
    # ---------------------------------------------------------
    v = sample['embedding']
    if torch.is_tensor(v): v = v.numpy()
    print(f"\n🔹 [Embedding] 形状: {v.shape}, 类型: {v.dtype}")
    
    # ---------------------------------------------------------
    # B. 检查 Meta (核心检查点)
    # ---------------------------------------------------------
    meta = sample['meta']
    if torch.is_tensor(meta): meta = meta.numpy()
    cx, cy, s = meta[0], meta[1], meta[2]
    N = meta[3] if len(meta) > 3 else "未知"
    print(f"\n🔹 [Meta 信息]")
    print(f"   -> 中心点坐标 (cx, cy): ({cx:.4f}, {cy:.4f})")
    print(f"   -> 尺度因子 (s): {s:.4f}")
    print(f"   -> 原始顶点数 (N): {N}")
    
    # ---------------------------------------------------------
    # C. 检查 Triangles (原始三角形坐标)
    # ---------------------------------------------------------
    tris_raw = sample['triangles']
    if torch.is_tensor(tris_raw): tris_raw = tris_raw.numpy()
    print(f"\n🔹 [Triangles 原始坐标] 形状: {tris_raw.shape}")
    
    # 将所有三角形的所有顶点展平，求包围盒
    verts = tris_raw.reshape(-1, 2)
    min_x, max_x = verts[:, 0].min(), verts[:, 0].max()
    min_y, max_y = verts[:, 1].min(), verts[:, 1].max()
    raw_width = max_x - min_x
    raw_height = max_y - min_y
    print(f"   -> 实际X范围:[{min_x:.4f}, {max_x:.4f}] (宽: {raw_width:.4f})")
    print(f"   -> 实际Y范围:[{min_y:.4f}, {max_y:.4f}] (高: {raw_height:.4f})")
    
    # ---------------------------------------------------------
    # D. 模拟 DataLoader 中的归一化，找出 [-0.5, 0.5] 的原因
    # ---------------------------------------------------------
    tris_norm_old = (tris_raw - np.array([cx, cy])) / (s + 1e-9)
    verts_norm_old = tris_norm_old.reshape(-1, 2)
    norm_min_x, norm_max_x = verts_norm_old[:, 0].min(), verts_norm_old[:, 0].max()
    norm_min_y, norm_max_y = verts_norm_old[:, 1].min(), verts_norm_old[:, 1].max()
    
    print("\n🚨[归一化公式复现] tris_norm = (tris_raw - center) / s")
    print(f"   -> 归一化后 X 范围:[{norm_min_x:.4f}, {norm_max_x:.4f}]")
    print(f"   -> 归一化后 Y 范围:[{norm_min_y:.4f}, {norm_max_y:.4f}]")
    
    # 诊断逻辑
    if abs(norm_max_x) <= 0.55 and abs(norm_max_y) <= 0.55:
        print("\n💡 阶段一诊断结论:")
        print("问题已定位！Meta 中的 's' 显然代表了包围盒的【全宽/全高】。")
        print("因为中心点到边缘的距离最大只有 s/2，如果你除以全宽 s，")
        print("结果自然会被压缩到 [-0.5, 0.5] 的区间内。")
        print("后续修改意见：在 loader 中将公式改为 `/ (s / 2.0)` 即可铺满 [-1, 1]。")

if __name__ == "__main__":
    # 请根据实际情况修改数据路径
    DATA_FILE = "data/encoded_samples_20260324_1910.pt" # 替换为你的文件名
    inspect_data(DATA_FILE)