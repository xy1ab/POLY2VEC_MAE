import os, torch, random, time, json
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from models import NaturalResourceFoundationModel
import argparse

# 🌟 1. 硬件自适应驱动
try:
    import torch_musa
except ImportError:
    pass

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_type = "musa" if (hasattr(torch, "musa") and torch.musa.is_available()) else "cuda"
    
    if device_type == "musa":
        torch.musa.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)
    return local_rank, device_type

def load_all_caches(cache_files, is_master):
    all_layers_data = {}
    for file in cache_files:
        if os.path.exists(file):
            if is_master: print(f"📦 正在装载高速缓存: {file} ...")
            data = torch.load(file, map_location='cpu', weights_only=False)
            all_layers_data.update(data)
    return all_layers_data

def train():
    local_rank, device_type = setup_ddp()
    is_master = (dist.get_rank() == 0)
    target_device = torch.device(device_type, local_rank)
    
    # 🌟 2. 0.1B 亿级模型规格 (768维, 12层, 12头)
    config = {
        'truth_dim': 256, 
        'semantic_dim': 768, 
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'vocab_size': 20000
    }
    
    batch_size, epochs, base_lr = 256, 2, 1e-4 # 冒烟测试参数
    
    model = NaturalResourceFoundationModel(config).to(target_device)
    if is_master:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"⚙️ 模型编译完毕！ZRZYB 0.1B 底座当前参数量: {total_params / 1e6:.2f} M")

    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    scaler = GradScaler(device_type)
    
    # 加载数据源
    cache_files = ["cache_aanp.pt", "cache_lincao.pt", "cache_fujian.pt"]
    all_data = load_all_caches(cache_files, is_master)
    layer_names = list(all_data.keys())
    
    if is_master: print(f"📊 数据就绪，正在启动异构 Batch 混洗引擎...")

    for epoch in range(epochs):
        random.shuffle(layer_names)
        epoch_loss = 0.0
        total_b = 0
        
        for lname in layer_names:
            layer = all_data[lname]
            n_samples = layer['meta']['total_samples']
            if n_samples < batch_size: continue
            
            # 🌟 3. 核心修复：精准对接 ZRZY版本1 的 6 路全息数据张量
            ds = TensorDataset(
                torch.from_numpy(layer['cont_int']),
                torch.from_numpy(layer['cont_frac_hi']),
                torch.from_numpy(layer['cont_frac_lo']),
                torch.from_numpy(layer['cont_norm']),
                torch.from_numpy(layer['word_data']),
                torch.from_numpy(layer['char_data'])
            )
            
            sm = DistributedSampler(ds, shuffle=True)
            sm.set_epoch(epoch)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sm, num_workers=4, pin_memory=True)
            
            for batch in loader:
                batch = [t.to(target_device, non_blocking=True) for t in batch]
                optimizer.zero_grad()
                
                # 🌟 4. 混合精度加速：对齐硬件
                with autocast(device_type=device_type):
                    # 传入全部 6 路数据，解开刚才的 TypeError
                    _, loss = model(*batch)
                    loss = loss.mean()
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                total_b += 1
            
        if is_master and total_b > 0:
            print(f"📈 Ep {epoch+1}/{epochs} | Avg Loss: {epoch_loss/total_b:.6f}")
            # 保存快照
            torch.save(model.module.state_dict(), f"./outputs/best_model_0.1B_v1.pth")

    if is_master: print("✅ 冒烟测试完成！模型已成功扩容并跑通 6 路全息数据。")

if __name__ == "__main__":
    train()