import os
import gc
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 尝试从 model.py 导入现有模型 (如果存在)
try:
    from model import GCN_CSS, CNN_CSS, MLP_CSS, GraphConv
except ImportError:
    pass

# ================= 配置区域 =================
HDF5_PATH = '/root/autodl-tmp/radioml2018/GCN_CSS/GOLD_XYZ_OSC.0001_1024.hdf5' 
BATCH_SIZE = 32  
NUM_NODES = 32
TARGET_PFA = 0.1 
SAMPLES_PER_SNR = 100 

# 使用 CPU 推理 (稳定且避免 OOM)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ==========================================
# 补充定义缺失的模型类 (确保独立运行)
# ==========================================

class GraphConv(layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.bn = layers.BatchNormalization()

    def build(self, input_shape):
        feature_shape = input_shape[0]
        input_dim = feature_shape[-1]
        self.w = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        features, adjacency = inputs
        support = tf.matmul(features, self.w)
        output = tf.matmul(adjacency, support) + self.b
        output = self.bn(output)
        if self.activation: output = self.activation(output)
        return output

class GraphAttention(layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(GraphAttention, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.bn = layers.BatchNormalization()

    def build(self, input_shape):
        feature_shape = input_shape[0]
        input_dim = feature_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.a = self.add_weight(shape=(2 * self.units, 1), initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        h, adj = inputs
        Wh = tf.matmul(h, self.W)
        N = tf.shape(Wh)[1]
        Wh_r1 = tf.repeat(tf.expand_dims(Wh, 2), N, axis=2)
        Wh_r2 = tf.repeat(tf.expand_dims(Wh, 1), N, axis=1)
        concat_Wh = tf.concat([Wh_r1, Wh_r2], axis=-1)
        e = tf.matmul(concat_Wh, self.a)
        e = tf.squeeze(e, -1)
        e = layers.LeakyReLU(alpha=0.2)(e)
        zero_vec = -9e15 * tf.ones_like(e)
        attention = tf.where(adj > 0, e, zero_vec)
        attention = tf.nn.softmax(attention, axis=-1)
        output = tf.matmul(attention, Wh)
        output = self.bn(output)
        if self.activation: output = self.activation(output)
        return output

# --- 模型定义 ---

class GCN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(GCN_CSS, self).__init__()
        self.gcn1 = GraphConv(32, activation='relu')
        self.gcn2 = GraphConv(64, activation='relu')
        self.gcn3 = GraphConv(128, activation='relu')
        self.pooling = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.gcn1([x, a]); x = self.gcn2([x, a]); x = self.gcn3([x, a])
        x = self.pooling(x); x = self.fc1(x); x = self.dropout(x)
        return self.fc2(x)

class CNN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(CNN_CSS, self).__init__()
        self.conv1 = layers.Conv1D(64, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization(); self.pool1 = layers.MaxPooling1D(2)
        self.conv2 = layers.Conv1D(128, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization(); self.pool2 = layers.MaxPooling1D(2)
        self.conv3 = layers.Conv1D(128, 3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.global_pool = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, _ = inputs
        x = tf.reshape(x, [-1, 1024, 2])
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.global_pool(x)
        x = self.dropout(self.fc1(x))
        return self.fc2(x)

class MLP_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(MLP_CSS, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu')
        self.bn1 = layers.BatchNormalization(); self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(512, activation='relu')
        self.bn2 = layers.BatchNormalization(); self.dropout2 = layers.Dropout(0.5)
        self.fc_out = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, _ = inputs
        x = self.flatten(x)
        x = self.dropout1(self.bn1(self.dense1(x)))
        x = self.dropout2(self.bn2(self.dense2(x)))
        return self.fc_out(x)

class GAT_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(GAT_CSS, self).__init__()
        self.gat1 = GraphAttention(32, activation='relu')
        self.gat2 = GraphAttention(64, activation='relu')
        self.gat3 = GraphAttention(128, activation='relu')
        self.pool = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.gat1([x, a]); x = self.gat2([x, a]); x = self.gat3([x, a])
        x = self.pool(x); x = self.fc1(x); x = self.dropout(x)
        return self.fc2(x)

class CNNGCN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(CNNGCN_CSS, self).__init__()
        self.conv1 = layers.Conv1D(32, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.gcn1 = GraphConv(64, activation='relu')
        self.gcn2 = GraphConv(128, activation='relu')
        self.pool = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.bn1(self.conv1(x))
        x = self.gcn1([x, a]); x = self.gcn2([x, a])
        x = self.pool(x); x = self.fc1(x); x = self.dropout(x)
        return self.fc2(x)

# ================= 模型列表配置 =================
MODELS = [
    {'name': 'GCN',     'class': GCN_CSS,    'path': 'best_gcn_model.h5',    'color': 'red',    'marker': 'o', 'type': 'graph'},
    {'name': 'CNN',     'class': CNN_CSS,    'path': 'best_cnn_model.h5',    'color': 'blue',   'marker': 's', 'type': 'flat'},
    {'name': 'MLP',     'class': MLP_CSS,    'path': 'best_mlp_model.h5',    'color': 'green',  'marker': '^', 'type': 'flat'},
    {'name': 'GAT',     'class': GAT_CSS,    'path': 'best_gat_model.h5',    'color': 'purple', 'marker': 'D', 'type': 'graph'},
    {'name': 'CNNGCN',  'class': CNNGCN_CSS, 'path': 'best_cnngcn_model.h5', 'color': 'orange', 'marker': 'x', 'type': 'graph'},
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
            
        idx_floor = noise_floor_indices[:2000]
        X_floor = f['X'][idx_floor]
        noise_std = np.std(X_floor)
        print(f"📉 估计底噪 Std: {noise_std:.6f}")

    X_noise = np.random.normal(0, noise_std, size=X_sig.shape).astype(np.float32)
    Z_noise = np.full((len(X_sig), 1), -100.0)
    
    X = np.concatenate([X_noise, X_sig], axis=0)
    Y = np.concatenate([np.zeros(len(X_sig)), np.ones(len(X_sig))])
    Z = np.concatenate([Z_noise, Z_sig])
    
    del X_chunks, X_sig, X_noise, Z_all, X_floor
    gc.collect()
    return X, Y, Z.flatten()

# ================= 数据预处理 (已修复) =================
def process_batch(X_raw, model_type='graph'):
    feat_dim = 1024 * 2 // NUM_NODES
    X_r = X_raw.reshape(-1, NUM_NODES, feat_dim)
    X_t = tf.convert_to_tensor(X_r, dtype=tf.float32)
    
    if model_type == 'graph':
        # 图神经网络需要邻接矩阵
        diff = tf.expand_dims(X_t, 2) - tf.expand_dims(X_t, 1)
        dist = tf.reduce_sum(tf.square(diff), axis=-1)
        A = tf.exp(-dist) 
        D = tf.reduce_sum(A, axis=-1, keepdims=True)
        A = A / (D + 1e-6)
        return [X_t, A]
    else:
        # 【关键修复】MLP/CNN 需要一个假的输入来占位，但 Batch Size 必须匹配
        batch_size = tf.shape(X_t)[0]
        dummy = tf.zeros((batch_size, 1), dtype=tf.float32)
        return [X_t, dummy]

# ================= 预测循环 =================
def get_predictions(model_cfg, X):
    print(f"🤖 评估模型: {model_cfg['name']}...")
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = model_cfg['class'](2, NUM_NODES)
    
    try:
        if model_cfg['type'] == 'graph':
            model.build([(None, NUM_NODES, 64), (None, NUM_NODES, NUM_NODES)])
        else:
            model.build([(None, NUM_NODES, 64), (None, 1)]) # 占位符 shape 匹配
            
        model.load_weights(model_cfg['path'])
    except Exception as e:
        print(f"❌ {model_cfg['name']} 权重加载失败: {e}")
        return None
        
    preds = []
    total = len(X)
    
    for i in range(0, total, BATCH_SIZE):
        bx = X[i : i+BATCH_SIZE]
        inputs = process_batch(bx, model_type=model_cfg['type'])
        p = model.predict_on_batch(inputs)
        preds.append(p[:, 1])
        
        if i % (BATCH_SIZE * 50) == 0:
            print(f"   {i}/{total}", end='\r')
            
    return np.concatenate(preds)

# ================= 绘图逻辑 =================
def plot_charts(results, Y_true, Z_snr):
    # 1. 检测概率 vs SNR
    plt.figure(figsize=(10, 6))
    snr_range = np.arange(-20, 7, 1) 
    
    for name, scores in results.items():
        cfg = next(c for c in MODELS if c['name'] == name)
        
        noise_scores = scores[Y_true == 0]
        thresh = np.percentile(noise_scores, (1 - TARGET_PFA)*100)
        
        pd_list = []
        valid_snrs = []
        for snr in snr_range:
            idx = np.where((Y_true == 1) & (np.abs(Z_snr - snr) < 0.5))[0]
            if len(idx) > 0:
                pd = np.mean(scores[idx] > thresh)
                pd_list.append(pd)
                valid_snrs.append(snr)
            
        plt.plot(valid_snrs, pd_list, label=name, color=cfg['color'], marker=cfg['marker'], markersize=4)
                 
    plt.title(f'Detection Probability vs SNR ($P_{{fa}}={TARGET_PFA}$)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Pd')
    plt.xlim([-20, 6])
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('pd_vs_snr_compare.png', dpi=300)
    print("\n✅ Pd 曲线已保存: pd_vs_snr_compare.png")

    # 2. ROC 曲线
    target_snrs = [-10, -8]
    for target_snr in target_snrs:
        plt.figure(figsize=(8, 8))
        has_data = False
        for name, scores in results.items():
            cfg = next(c for c in MODELS if c['name'] == name)
            
            sig_idx = np.where((Y_true == 1) & (np.abs(Z_snr - target_snr) < 0.5))[0]
            noise_idx = np.where(Y_true == 0)[0]
            
            if len(sig_idx) > 0:
                has_data = True
                y_sub = np.concatenate([np.zeros(len(noise_idx)), np.ones(len(sig_idx))])
                s_sub = np.concatenate([scores[noise_idx], scores[sig_idx]])
                
                fpr, tpr, _ = roc_curve(y_sub, s_sub)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', color=cfg['color'])
        
        if has_data:
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'ROC Curve at SNR = {target_snr}dB')
            plt.xlabel('FPR'); plt.ylabel('TPR')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(f'roc_curve_{target_snr}dB.png', dpi=300)
            print(f"✅ ROC 曲线已保存: roc_curve_{target_snr}dB.png")

if __name__ == "__main__":
    if not os.path.exists(HDF5_PATH):
        print(f"❌ 错误: 数据集文件不存在: {HDF5_PATH}")
    else:
        X, Y, Z = load_random_test_data(HDF5_PATH, samples_per_snr=SAMPLES_PER_SNR)
        
        results = {}
        for m in MODELS:
            if os.path.exists(m['path']):
                s = get_predictions(m, X)
                if s is not None: results[m['name']] = s
            else:
                print(f"⚠️ 跳过 {m['name']}: 权重文件不存在")
                
        if results:
            plot_charts(results, Y, Z)
        else:
            print("❌ 没有产生任何结果，请检查权重文件。")