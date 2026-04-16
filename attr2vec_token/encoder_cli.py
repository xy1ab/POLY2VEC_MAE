import os, glob, torch, argparse, pyogrio, warnings
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from config import ModelConfig
from models import NaturalResourceFoundationModel

# 🌟 屏蔽底层的空间数据类型转换警告，保持控制台整洁
warnings.filterwarnings('ignore', category=UserWarning)

MUST_STR = ('代码', '编码', '编号', 'id', 'code', 'bm', 'dm', 'bsm', 'pac', 'politcode')

def get_safe_dtype(csv_path):
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns
        return {c: str for c in cols if any(s in c.lower() for s in MUST_STR)}
    except:
        return None

def float64_to_three_float32(arr):
    arr = np.nan_to_num(arr, nan=0.0) 
    int_part = np.trunc(arr).astype(np.float32)
    frac = arr - int_part
    f_hi = np.trunc(frac * 10000).astype(np.float32)
    f_lo = np.trunc((frac * 10000 - f_hi) * 10000).astype(np.float32)
    return int_part, f_hi, f_lo

class EncoderMachine:
    def __init__(self, config_path):
        torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
        self.config = ModelConfig(); self.config.load(config_path)
        self.tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NaturalResourceFoundationModel(self.config).to(self.device)
        self.model.eval()

    def process(self, df):
        full_cols = [c for c in df.columns if c.lower() not in ['geometry', 'shape']]
        num_cols = [c for c in full_cols if pd.api.types.is_numeric_dtype(df[c])]
        str_cols = [c for c in full_cols if c not in num_cols]
        N = len(df)
        
        # 🛡️ 异常防御：如果遇到没有任何数据的空图层，直接返回空张量
        if N == 0:
            return torch.zeros((0, self.config.truth_dim), dtype=torch.float32), num_cols, str_cols, full_cols

        # 1. 数值处理
        num_features = np.zeros((N, len(num_cols) * 3), dtype=np.float32)
        if num_cols:
            int_p, fh, fl = float64_to_three_float32(df[num_cols].values.astype(np.float64))
            for i in range(len(num_cols)):
                num_features[:, i*3:i*3+3] = np.stack([int_p[:,i], fh[:,i], fl[:,i]], axis=1)

        # 2. 文本处理
        seq_ids = np.zeros((N, self.config.max_seq_len), dtype=np.float32)
        if str_cols:
            # 🌟 核心修复：弃用 Pandas 的 .agg，改用底层 values 遍历，杜绝类型推断 Bug
            combined = [' [SEP] '.join(row) for row in df[str_cols].fillna("").astype(str).values]
            encoded = self.tokenizer.encode_batch(combined)
            for i, enc in enumerate(encoded):
                ids = enc.ids[:self.config.max_seq_len]
                seq_ids[i, :len(ids)] = ids

        # 3. 张量拼接
        truth_vec = np.zeros((N, self.config.truth_dim), dtype=np.float32)
        truth_vec[:, :num_features.shape[1]] = num_features
        truth_vec[:, num_features.shape[1] : num_features.shape[1] + self.config.max_seq_len] = seq_ids
        
        with torch.no_grad():
            emb = self.model.inn_core(torch.tensor(truth_vec).to(self.device), reverse=False)
        return emb.cpu(), num_cols, str_cols, full_cols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="physical_assets.pt")
    args = parser.parse_args()

    worker = EncoderMachine(args.config)
    bundle = []
    
    for f in glob.glob(os.path.join(args.data_dir, "*")):
        ext = os.path.splitext(f)[1].lower()
        if ext == '.csv':
            try:
                name = os.path.basename(f).split('.')[0]
                df = pd.read_csv(f, nrows=5, dtype=get_safe_dtype(f))
                emb, n_c, s_c, f_c = worker.process(df)
                bundle.append({"name": name, "emb": emb, "num_cols": n_c, "str_cols": s_c, "full_order": f_c, "type": "CSV", "path": f})
            except Exception as e:
                print(f"⚠️ 跳过 CSV {f}: {e}")
        elif ext == '.gdb':
            try:
                for layer_name, _ in pyogrio.list_layers(f):
                    try:
                        df = pyogrio.read_dataframe(f, layer=layer_name, max_features=5, read_geometry=False)
                        emb, n_c, s_c, f_c = worker.process(df)
                        bundle.append({"name": layer_name, "emb": emb, "num_cols": n_c, "str_cols": s_c, "full_order": f_c, "type": "GDB", "path": f})
                    except Exception as e:
                        print(f"⚠️ 跳过 GDB图层 {layer_name}: {e}")
            except: pass

    torch.save(bundle, args.out)
    print(f"✅ 编码完成，共成功打包 {len(bundle)} 个图层。")