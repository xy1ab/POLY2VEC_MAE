# config.py
class FourierConfig:
    # --- 傅里叶引擎参数 (对齐 run_cft_visualization.py) ---
    POS_FREQS = 31
    W_MIN = 0.1
    W_MAX = 100.0         
    FREQ_TYPE = 'geometric' # 几何级数，低频密高频疏
    PATCH_SIZE = 16
    
    # --- 空间域参数 ---
    SPATIAL_SIZE = 256      # 图像分辨率