import os
import gc
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 导入自定义模型
from model import GCN_CSS, CNN_CSS, MLP_CSS 

# ================= 配置区域 =================
HDF5_PATH = '/root/autodl-tmp/radioml2018/GCN_CSS/GOLD_XYZ_OSC.0001_1024.hdf5' 
BATCH_SIZE = 32  
NUM_NODES = 32
TARGET_PFA = 0.1 
SAMPLES_PER_SNR = 100 # 恢复采样数，保证曲线平滑

# 强制使用 CPU (避免 GPU OOM，虽然慢点但稳)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODELS = [
    {'name': 'GCN-CSS (Proposed)', 'class': GCN_CSS, 'path': 'best_gcn_model.h5', 'color': 'red', 'marker': 'o', 'type': 'gcn'},
    {'name': 'CNN', 'class': CNN_CSS, 'path': 'best_cnn_model.h5', 'color': 'blue', 'marker': 's', 'type': 'other'},
    {'name': 'MLP', 'class': MLP_CSS, 'path': 'best_mlp_model.h5', 'color': 'green', 'marker': '^', 'type': 'other'},
]

# ================= 数据加载 (修复版) =================
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
        
        # 【关键修正】: 准确估算底噪 (Noise Floor)
        # 使用 -20dB 的信号作为底噪参考 (此时信号淹没在噪声中，接近纯噪声)
        noise_floor_indices = np.where(Z_all == -20)[0]
        if len(noise_floor_indices) == 0:
            # 如果没有 -20dB，找最小的那个 SNR
            min_snr = np.min(Z_all)
            noise_floor_indices = np.where(Z_all == min_snr)[0]
            print(f"⚠️ 未找到 -20dB 数据，使用 {min_snr}dB 估算底噪")
            
        # 只取前 2000 个样本计算 std，节省内存
        idx_floor = noise_floor_indices[:2000]
        # 需要重新从文件读取这部分纯底噪数据
        X_floor = f['X'][idx_floor]
        noise_std = np.std(X_floor)
        print(f"📉 估计的物理底噪 Std: {noise_std:.6f}")

    # 生成 H0 噪声
    # 这里的噪声功率必须与数据集的底噪一致，模型才能正确区分
    X_noise = np.random.normal(0, noise_std, size=X_sig.shape).astype(np.float32)
    Z_noise = np.full((len(X_sig), 1), -100.0)
    
    X = np.concatenate([X_noise, X_sig], axis=0)
    Y = np.concatenate([np.zeros(len(X_sig)), np.ones(len(X_sig))])
    Z = np.concatenate([Z_noise, Z_sig])
    
    # 【重要】删除了 Z-Score 归一化！
    # 保持 X 的原始幅度，因为训练时并未归一化
    
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
        # GCN 计算邻接矩阵
        diff = tf.expand_dims(X_t, 2) - tf.expand_dims(X_t, 1)
        dist = tf.reduce_sum(tf.square(diff), axis=-1)
        A = tf.exp(-dist) 
        D = tf.reduce_sum(A, axis=-1, keepdims=True)
        A = A / (D + 1e-6)
        return [X_t, A]
    else:
        # CNN/MLP 传 Dummy Tensor
        batch_size = tf.shape(X_t)[0]
        dummy = tf.zeros((batch_size, 1), dtype=tf.float32)
        return [X_t, dummy]

def get_predictions(model_cfg, X):
    print(f"🤖 正在评估: {model_cfg['name']}...")
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = model_cfg['class'](2, NUM_NODES)
    try:
        model.build([(None, NUM_NODES, 64), (None, NUM_NODES, NUM_NODES)])
        model.load_weights(model_cfg['path'])
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
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

def plot_charts(results, Y_true, Z_snr):
    # 图 1: Pd vs SNR
    plt.figure(figsize=(10, 6))
    snr_range = np.arange(-20, 31, 2)
    
    for name, scores in results.items():
        cfg = next(c for c in MODELS if c['name'] == name)
        
        # 计算阈值
        noise_scores = scores[Y_true == 0]
        thresh = np.percentile(noise_scores, (1 - TARGET_PFA)*100)
        
        pd_list = []
        for snr in snr_range:
            idx = np.where((Y_true == 1) & (np.abs(Z_snr - snr) < 1.0))[0]
            if len(idx) == 0: 
                pd_list.append(0)
            else:
                pd = np.mean(scores[idx] > thresh)
                pd_list.append(pd)
            
        plt.plot(snr_range, pd_list, label=name, color=cfg['color'], marker=cfg['marker'])
                 
    plt.title(f'Detection Probability vs SNR ($P_{{fa}}={TARGET_PFA}$)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Pd')
    plt.xlim([-20, 30])
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.savefig('real_pd_vs_snr_fixed.png')
    print("✅ 图1 保存成功: real_pd_vs_snr_fixed.png")

    # 图 2: ROC
    plt.figure(figsize=(8, 8))
    target_snr = -10
    sig_idx = np.where((Y_true == 1) & (np.abs(Z_snr - target_snr) < 1.0))[0]
    noise_idx = np.where(Y_true == 0)[0]
    
    if len(sig_idx) > 0:
        y_roc = np.concatenate([np.zeros(len(noise_idx)), np.ones(len(sig_idx))])
        for name, scores in results.items():
            cfg = next(c for c in MODELS if c['name'] == name)
            s_roc = np.concatenate([scores[noise_idx], scores[sig_idx]])
            fpr, tpr, _ = roc_curve(y_roc, s_roc)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})", color=cfg['color'])
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC at {target_snr}dB')
    plt.legend()
    plt.savefig('real_roc_curve_fixed.png')
    print("✅ 图2 保存成功: real_roc_curve_fixed.png")

if __name__ == "__main__":
    X, Y, Z = load_random_test_data(HDF5_PATH, samples_per_snr=SAMPLES_PER_SNR)
    
    results = {}
    for m in MODELS:
        s = get_predictions(m, X)
        if s is not None: results[m['name']] = s
            
    if results: plot_charts(results, Y, Z)