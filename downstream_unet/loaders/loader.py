# loaders/loader.py
import torch
import gc
import os
from torch.utils.data import Dataset
from pathlib import Path
import bisect
import random
from torch.utils.data import Sampler
import math
import torch.distributed as dist

class ShardDataset(Dataset):
    def __init__(self, data_dir):
        """
        初始化分片数据集
        :param data_dir: 包含 shard_000x.pt 文件的目录
        """
        self.data_dir = Path(data_dir)
        
        # 1. 严格筛选数据分片文件
        self.shard_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        
        if not self.shard_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到分片文件")

        # 2. 元数据隔离
        self.meta_path = self.data_dir.parent / "metadata.pt"
        
        # 3. 获取/扫描 shard_sizes
        needs_rebuild = False
        if self.meta_path.exists():
            try:
                self.shard_sizes = torch.load(self.meta_path, map_location='cpu')
                if len(self.shard_sizes) != len(self.shard_files):
                    needs_rebuild = True
            except:
                needs_rebuild = True
        else:
            needs_rebuild = True

        if needs_rebuild:
            print(f"🔍 正在扫描分片生成元数据: {self.meta_path}")
            self.shard_sizes = []
            for f in self.shard_files:
                tmp_data = torch.load(self.data_dir / f, map_location='cpu', weights_only=True)
                self.shard_sizes.append(len(tmp_data))
                del tmp_data
            torch.save(self.shard_sizes, self.meta_path)

        # 4. 计算全局累积偏移量表
        self.cumsum = [0]
        for s in self.shard_sizes:
            self.cumsum.append(self.cumsum[-1] + s)

        # 修正 Bug 1: 定义 self.total_len，否则 __len__ 会报错
        self.total_len = self.cumsum[-1]

        # 5. 初始化运行时缓存状态
        # 修正 Bug 2: 变量名必须与 _load_shard 内部一致
        self.current_shard_data = None 
        self.current_shard_idx = -1
    
    def _load_shard(self, shard_idx):
        if self.current_shard_idx == shard_idx:
            return
        
        # 释放内存
        self.current_shard_data = None 
        gc.collect() 
        
        # 修正 Bug 3: shard_files 存的是文件名字符串，load 时需拼接路径
        shard_name = self.shard_files[shard_idx]
        print(f"\n[IO] Loading shard: {shard_name}")
        self.current_shard_data = torch.load(self.data_dir / shard_name, map_location='cpu')
        self.current_shard_idx = shard_idx
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # 使用二分查找定位分片
        shard_idx = bisect.bisect_right(self.cumsum, idx) - 1
        local_idx = idx - self.cumsum[shard_idx]
        self._load_shard(shard_idx)
        
        sample = self.current_shard_data[local_idx]
        return sample['input'].float(), sample['label'].float()
    
class DistributedShardSampler(Sampler):
    def __init__(self, shard_sizes, batch_size, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
            else:
                try:
                    num_replicas = dist.get_world_size()
                except RuntimeError:
                    num_replicas = 1
        if rank is None:
            if not dist.is_available():
                rank = 0
            else:
                try:
                    rank = dist.get_rank()
                except RuntimeError:
                    rank = 0

        self.shard_sizes = shard_sizes
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        
        self.cumsum = [0]
        for s in shard_sizes:
            self.cumsum.append(self.cumsum[-1] + s)
        
        # 核心逻辑：将总索引按 rank 分配
        self.total_size = self.cumsum[-1]
        # 确保每个进程分配到的样本数一致（不足补齐，防止 DDP 等待死锁）
        self.num_samples = math.ceil(self.total_size / self.num_replicas)
        self.total_padded_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 1. 生成所有分片的全局索引
        # 我们依然按分片遍历，但每个分片内只取属于当前 rank 的索引
        g = torch.Generator()
        g.manual_seed(getattr(self, 'epoch', 0)) # 联动 set_epoch

        shard_indices = list(range(len(self.shard_sizes)))
        if self.shuffle:
            random.seed(getattr(self, 'epoch', 0))
            random.shuffle(shard_indices)

        indices = []
        for s_idx in shard_indices:
            start = self.cumsum[s_idx]
            end = self.cumsum[s_idx + 1]
            shard_indices_list = list(range(start, end))
            
            if self.shuffle:
                random.shuffle(shard_indices_list)
            
            indices.extend(shard_indices_list)

        # 2. 补齐数据使之能被 world_size 整除
        indices += indices[:(self.total_padded_size - len(indices))]
        
        # 3. 核心分片：当前 rank 只取自己的那一部分
        # 这样可以保证每个 rank 负责的索引虽然跨分片，但读取时依然具有一定的局部性
        offset = self.rank * self.num_samples
        mine = indices[offset : offset + self.num_samples]

        # 4. 按 Batch 输出
        for i in range(0, len(mine), self.batch_size):
            yield mine[i : i + self.batch_size]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

class SingleDataset(Dataset):
    """单文件数据集（用于验证/测试）"""
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"文件不存在: {data_path}")
        self.data = torch.load(self.data_path, map_location='cpu', weights_only=False)
        print(f"📊 加载 {data_path}: {len(self.data)} 样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['input'].float(), sample['label'].float()