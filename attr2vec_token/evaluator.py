import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import json
import glob
import numpy as np
from models import NaturalResourceFoundationModel
from tokenizers import Tokenizer 

# ==========================================
# 1. 数据审计员（负责对账单渲染）
# ==========================================
class DataAuditor:
    def __init__(self, tokenizer, vocab): 
        self.tokenizer = tokenizer

    def render_section(self, source, layer, samples, mode="text"):
        if not samples: return
        print(f"\n{'='*130}\n📊 数据源: {source:<15} | 图层: {layer:<25}")
        print(f"📌 模式: {'🔤 文本一致性对账' if mode=='text' else '📈 连续数值高精度对账'} (展示 {len(samples)} 条)")
        print(f"{'Idx':<4} | {'输入底座的真实原始数据 (Ground Truth)':<60} | {'大模型物理无损解码结果 (Decoded)':<60}\n{'-'*130}")
        for i, (orig, dec) in enumerate(samples):
            print(f"{i+1:<4} | {str(orig):<60} | {str(dec):<60}")
        print(f"{'='*130}")

# ==========================================
# 2. 核心评估流水线
# ==========================================
def evaluate():
    print("🔬 [大一统底座] 全要素无损审计系统启动...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists("zrzy_tokenizer.json"):
        raise FileNotFoundError("❌ 未找到 zrzy_tokenizer.json 分词器，请先运行 train_tokenizer_modify.py")
    tokenizer = Tokenizer.from_file("zrzy_tokenizer.json")
    
    vocab_path = "global_vocab_auto.json"
    vocab = json.load(open(vocab_path, "r", encoding="utf-8")) if os.path.exists(vocab_path) else {}
    auditor = DataAuditor(tokenizer, vocab)

    config = {
        'truth_dim': 256,
        'semantic_dim': 256,
        'vocab_size': 20000,
        'max_seq_len': 64
    }
    model = NaturalResourceFoundationModel(config).to(device)
    
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到权重文件 {model_path}，请先完成训练。")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print("✅ 成功装载最优生产权重 [best_model.pth]")

    cache_files = glob.glob("cache_*.pt")
    if not cache_files:
        print("⚠️ 未找到任何 cache_*.pt 数据缓存文件！")
        return

    latent_pool = None 

    with torch.no_grad():
        for cache_file in cache_files:
            source_name = os.path.basename(cache_file).replace("cache_", "").replace(".pt", "")
            data_dict = torch.load(cache_file, map_location=device, weights_only=False)
            
            for layer_name, layer_data in data_dict.items():
                meta = layer_data.get('meta', {})
                cont_cols = meta.get('cont_cols', [])
                char_cols = meta.get('char_cols', [])
                word_cols = meta.get('word_cols', [])
                max_seq_len = meta.get('max_seq_len', 64)
                
                num_cont, num_word, num_char = len(cont_cols), len(word_cols), len(char_cols)
                if num_cont == 0 and num_word == 0 and num_char == 0: continue

                TEST_BSZ = 64
                b0 = torch.tensor(layer_data['cont_int'][:TEST_BSZ], device=device)
                b1 = torch.tensor(layer_data['cont_frac_hi'][:TEST_BSZ], device=device)
                b2 = torch.tensor(layer_data['cont_frac_lo'][:TEST_BSZ], device=device)
                b3 = torch.tensor(layer_data['cont_norm'][:TEST_BSZ], device=device)
                b4 = torch.tensor(layer_data['word_data'][:TEST_BSZ], device=device)
                b5 = torch.tensor(layer_data['char_data'][:TEST_BSZ], device=device)

                latent, _ = model(b0, b1, b2, b3, b4, b5)
                if latent_pool is None: latent_pool = latent.cpu().numpy()[:40, :] 
                
                v_attr = latent[:, :config['truth_dim']]
                dec = model.inn_core(v_attr, reverse=True)
                
                c_exp = dec[:, :num_cont]
                c_hi = dec[:, num_cont:2*num_cont]
                c_lo = dec[:, 2*num_cont:3*num_cont]
                ch_sc = dec[:, 3*num_cont+num_word : 3*num_cont+num_word+(num_char*max_seq_len)]

                limit = min(5, b0.size(0))
                
                # ==========================================
                # 1. 连续数值对账 
                # ==========================================
                if num_cont > 0:
                    gt_cont = torch.ldexp(b1.double() + b2.double(), b0.int())
                    orig_cont = torch.ldexp(c_hi.double() + c_lo.double(), c_exp.int())
                    
                    cont_samples = []
                    for i in range(limit):
                        col_idx = 0 
                        gt_val = f"{gt_cont[i, col_idx].item():.6f}"
                        pred_val = f"{orig_cont[i, col_idx].item():.6f}"
                        cont_samples.append((f"[{cont_cols[col_idx]}] 原始真值: {gt_val}", f"模型预测: {pred_val}"))
                    auditor.render_section(source_name, layer_name, cont_samples, mode="num")

                # ==========================================
                # 2. 文本对账 
                # ==========================================
                if num_char > 0:
                    ch_sc = ch_sc.view(-1, num_char, max_seq_len)
                    
                    # 🌟 修复点：将输入的真值 (b5) 也乘回 32768.0 并转成整数
                    gt_char_ids = torch.round(b5 * 32768.0).long()
                    pred_char_ids = torch.round(ch_sc * 32768.0).long()
                    
                    char_samples = []
                    for i in range(limit):
                        col_idx = 0 
                        
                        # 拿着输入的真实 ID 去解出最初的汉字
                        gt_ids = gt_char_ids[i, col_idx].cpu().tolist()
                        gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True).replace(" ", "")
                        if not gt_text: gt_text = "[空]"
                        
                        # 拿着模型猜出来的 ID 去解汉字
                        dec_ids = pred_char_ids[i, col_idx].cpu().tolist()
                        dec_text = tokenizer.decode(dec_ids, skip_special_tokens=True).replace(" ", "")
                        if not dec_text: dec_text = "[解析失败/模型猜想为空]"
                        
                        char_samples.append((f"[{char_cols[col_idx]}] {gt_text}", f"{dec_text}"))
                    auditor.render_section(source_name, layer_name, char_samples, mode="text")

    if latent_pool is not None:
        print("\n🎨 对账完毕，正在渲染极客风报告图表...")
        plt.style.use('dark_background') 
        
        if os.path.exists("train_history.json"):
            with open("train_history.json", "r") as f: h = json.load(f)
            plt.figure(figsize=(12, 6))
            epochs = np.arange(len(h['loss']))
            losses = np.array(h['loss'])
            plt.plot(epochs, losses, color='#00ffcc', linewidth=2.5, label='Self-Supervised Reconstruction Loss')
            plt.fill_between(epochs, losses, color='#00ffcc', alpha=0.15)
            plt.title("Global Semantic Distillation Convergence", fontsize=18, color='white', pad=20)
            plt.xlabel("Epochs", fontsize=12, color='lightgray'); plt.ylabel("Loss", fontsize=12, color='lightgray')
            plt.grid(color='#333333', linestyle='--', linewidth=0.5)
            plt.legend(loc="upper right", facecolor='#111111', edgecolor='none')
            plt.savefig("vis_loss_curve.png", dpi=300, facecolor='#111111', bbox_inches='tight')

        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(latent_pool, cmap='mako', cbar_kws={'label': 'Energy Activation'})
        plt.title("Multi-Modal Latent Space Cross-Fusion (Batch: 40)", fontsize=18, color='white', pad=20)
        plt.xlabel(f"Latent Dimensions ({config['truth_dim']} Truth + {config['semantic_dim']} Semantic)", fontsize=12, color='lightgray')
        plt.ylabel("Batch Samples", fontsize=12, color='lightgray')
        plt.savefig("vis_latent_space.png", dpi=300, facecolor='#111111', bbox_inches='tight')
        print("✅ 报告图表已保存: vis_loss_curve.png, vis_latent_space.png")

if __name__ == "__main__":
    evaluate()