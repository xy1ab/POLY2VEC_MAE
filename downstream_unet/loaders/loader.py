# loaders/loader.py
import torch
import os
from torch.utils.data import Dataset

class V2Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path, map_location='cpu')
        # 读取时转回 float32 供模型计算
        return data['input'].float(), data['label'].float()