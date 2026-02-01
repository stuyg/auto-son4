import os
import gc
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 导入自定义模型
from model import GCN_CSS, CNN_CSS, MLP_CSS,GAT_CSS,CNN_GCN_CSS

# ================= 配置区域 =================
# 请确保此处路径正确
HDF5_PATH = '/root/autodl-tmp/radioml2018/GCN_CSS/GOLD_XYZ_OSC.0001_1024.hdf5' 
BATCH_SIZE = 32  
NUM_NODES = 32
TARGET_PFA = 0.1 
SAMPLES_PER_SNR = 1000 # 如果显存充足，可以适当调大

# 强制使用 CPU (避免 GPU OOM)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 确保这里的模型权重文件真实存在
MODELS = [
    {'name': 'GCN-CSS (Proposed)', 'class': GCN_CSS, 'path': 'best_gcn_model.h5', 'color': 'red', 'marker': 'o', 'type': 'gcn'},
    {'name': 'CNN', 'class': CNN_CSS, 'path': 'best_cnn_model.h5', 'color': 'blue', 'marker': 's', 'type': 'other'},
    {'name': 'MLP', 'class': MLP_CSS, 'path': 'best_mlp_model.h5', 'color': 'green', 'marker': '^', 'type': 'other'},
    {'name': 'GAT-Attn (New)', 'class': GAT_CSS, 'path': 'best_gat_model.h5', 'color': 'purple', 'marker': '*', 'type': 'gcn'},
    {'name': 'CNN-GCN (Hybrid)', 'class': CNN_GCN_CSS, 'path': 'best_cnngcn_model.h5', 'color': 'magenta', 'marker': '*', 'type': 'gcn'}
]

# ================= 数据加载 =================
def load_random_test_data(hdf5_path, samples_per_snr=100):
    print(f"🚀 正在加载测试数据 (每SNR采样: {samples_per_snr})...")
    with h5py.File(hdf5_path, 'r') as f:
        Z_all = f['Z'][:]
        unique_snrs = np.unique(Z_all)
        
        selected_indices = []
        np.random.seed(2024)
        for snr in unique_snrs:
            indices = np.where(Z_all == snr)[0]
            if len(indices) > samples_per_snr:
                chosen = np.random.choice(indices, samples_per_snr, replace=False)
            else:
                chosen = indices
            selected_indices.extend(chosen)
        selected_indices = np.sort(np.array(selected_indices))
        
        # 分块读取 X
        X_chunks = []
        chunk_size = 2000 
        for i in range(0, len(selected_indices), chunk_size):
            subset = selected_indices[i : i + chunk_size]
            X_chunks.append(f['X'][subset])
        
        X_sig = np.concatenate(X_chunks, axis=0)
        Z_sig = Z_all[selected_indices]
        
        # 估算底噪
        noise_floor_indices = np.where(Z_all == -20)[0]
        if len(noise_floor_indices) == 0:
            min_snr = np.min(Z_all)
            noise_floor_indices = np.where(Z_all == min_snr)[0]
            print(f"⚠️ 未找到 -20dB 数据，使用 {min_snr}dB 估算底噪")
            
        idx_floor = noise_floor_indices[:2000]
        X_floor = f['X'][idx_floor]
        noise_std = np.std(X_floor)
        print(f"📉 估计的物理底噪 Std: {noise_std:.6f}")

    # 生成 H0 噪声
    X_noise = np.random.normal(0, noise_std, size=X_sig.shape).astype(np.float32)
    Z_noise = np.full((len(X_sig), 1), -100.0)
    
    X = np.concatenate([X_noise, X_sig], axis=0)
    Y = np.concatenate([np.zeros(len(X_sig)), np.ones(len(X_sig))])
    Z = np.concatenate([Z_noise, Z_sig])
    
    del X_chunks, X_sig, X_noise, Z_all, X_floor
    gc.collect()
    
    print(f"✅ 数据就绪: {X.shape}")
    return X, Y, Z.flatten()

# ================= 批处理 =================
def process_batch(X_raw, is_gcn=True):
    feat_dim = 1024 * 2 // NUM_NODES
    X_r = X_raw.reshape(-1, NUM_NODES, feat_dim)
    X_t = tf.convert_to_tensor(X_r, dtype=tf.float32)
    
    if is_gcn:
        diff = tf.expand_dims(X_t, 2) - tf.expand_dims(X_t, 1)
        dist = tf.reduce_sum(tf.square(diff), axis=-1)
        A = tf.exp(-dist) 
        D = tf.reduce_sum(A, axis=-1, keepdims=True)
        A = A / (D + 1e-6)
        return [X_t, A]
    else:
        batch_size = tf.shape(X_t)[0]
        dummy = tf.zeros((batch_size, 1), dtype=tf.float32)
        return [X_t, dummy]

def get_predictions(model_cfg, X):
    print(f"🤖 正在评估: {model_cfg['name']}...")
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = model_cfg['class'](2, NUM_NODES)
    try:
        # Build 形状
        model.build([(None, NUM_NODES, 1024*2//NUM_NODES), (None, NUM_NODES, NUM_NODES)])
        model.load_weights(model_cfg['path'])
    except Exception as e:
        print(f"❌ 权重加载失败 ({model_cfg['path']}): {e}")
        return None
        
    preds = []
    total = len(X)
    is_gcn = (model_cfg['type'] == 'gcn')
    
    for i in range(0, total, BATCH_SIZE):
        bx = X[i : i+BATCH_SIZE]
        inputs = process_batch(bx, is_gcn=is_gcn)
        p = model.predict_on_batch(inputs)
        preds.append(p[:, 1])
        
        if i % (BATCH_SIZE * 50) == 0:
            print(f"   进度: {i}/{total}", end='\r')
            gc.collect()
            
    print(f"   进度: {total}/{total}")
    return np.concatenate(preds)

# ================= 绘图函数 (已修改) =================
def plot_charts(results, Y_true, Z_snr):
    # ------------------------------------------------
    # 1. Pd vs SNR (-20 to 5 dB)
    # ------------------------------------------------
    print("\n📊 正在绘制 Pd vs SNR (-20 ~ 5 dB)...")
    plt.figure(figsize=(10, 6))
    
    # 动态获取数据中存在的 SNR，并限制在 [-20, 5] 范围内
    all_snrs = np.unique(Z_snr)
    snr_range = [s for s in all_snrs if -20 <= s <= 5]
    snr_range.sort()
    
    for name, scores in results.items():
        cfg = next(c for c in MODELS if c['name'] == name)
        
        # 计算阈值
        noise_scores = scores[Y_true == 0]
        thresh = np.percentile(noise_scores, (1 - TARGET_PFA)*100)
        
        pd_list = []
        for snr in snr_range:
            # 容差设为 0.5 避免浮点精度问题
            idx = np.where((Y_true == 1) & (np.abs(Z_snr - snr) < 0.5))[0]
            if len(idx) == 0: 
                pd_list.append(0)
            else:
                pd = np.mean(scores[idx] > thresh)
                pd_list.append(pd)
            
        plt.plot(snr_range, pd_list, label=name, color=cfg['color'], marker=cfg['marker'])
                 
    plt.title(f'Detection Probability vs SNR ($P_{{fa}}={TARGET_PFA}$)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Pd')
    plt.xlim([-21, 6]) # 稍微放宽坐标轴以便看清边界
    plt.ylim([0, 1.05])
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.savefig('pd_vs_snr_limited.png')
    print("✅ 图1 保存成功: pd_vs_snr_limited.png")

    # ------------------------------------------------
    # 2. ROC Curves (-10dB 和 -8dB)
    # ------------------------------------------------
    target_snrs = [-10, -8]
    
    for target_snr in target_snrs:
        print(f"📊 正在绘制 ROC at {target_snr}dB...")
        plt.figure(figsize=(8, 8))
        
        sig_idx = np.where((Y_true == 1) & (np.abs(Z_snr - target_snr) < 0.5))[0]
        noise_idx = np.where(Y_true == 0)[0]
        
        if len(sig_idx) > 0:
            y_roc = np.concatenate([np.zeros(len(noise_idx)), np.ones(len(sig_idx))])
            for name, scores in results.items():
                cfg = next(c for c in MODELS if c['name'] == name)
                s_roc = np.concatenate([scores[noise_idx], scores[sig_idx]])
                
                fpr, tpr, _ = roc_curve(y_roc, s_roc)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})", color=cfg['color'])
        else:
            print(f"⚠️ 警告: 数据集中未找到 {target_snr}dB 的信号样本！")
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC at {target_snr}dB')
        plt.xlabel('False Positive Rate (Pfa)')
        plt.ylabel('True Positive Rate (Pd)')
        plt.legend()
        plt.grid(True)
        
        filename = f'roc_curve_{target_snr}dB.png'
        plt.savefig(filename)
        print(f"✅ 图2 保存成功: {filename}")

if __name__ == "__main__":
    # 加载数据
    X, Y, Z = load_random_test_data(HDF5_PATH, samples_per_snr=SAMPLES_PER_SNR)
    
    results = {}
    for m in MODELS:
        s = get_predictions(m, X)
        if s is not None: results[m['name']] = s
            
    if results: 
        plot_charts(results, Y, Z)
    else:
        print("❌ 没有产生任何预测结果，请检查模型权重文件是否存在。")