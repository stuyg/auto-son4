import h5py
import numpy as np
import tensorflow as tf
import math

class RadioMLSequence(tf.keras.utils.Sequence):
    def __init__(self, hdf5_path, batch_size, indices, num_antennas=2, mode='binary', fading='rayleigh'):
        """
        num_antennas (M): 天线数量。图的节点数将等于此值。
        fading: 'rayleigh' (瑞利), 'rician' (莱斯), 'awgn' (无衰落)
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.indices = indices
        self.num_nodes = num_antennas  # 这里的节点即天线
        self.mode = mode
        self.fading = fading
        self.num_classes = 2 if mode == 'binary' else 24
        
        # 每个天线接收完整的 1024 个 IQ 样本 (1024 * 2 = 2048 特征)
        self.feature_dim = 2048 
        
        print(f"📡 初始化多天线生成器: M={self.num_nodes} Antennas, Fading={fading}...")
        
        # 预加载底噪信息
        with h5py.File(self.hdf5_path, 'r') as f:
            # 简单估算数据集的底噪 (假设前1000个样本包含噪声)
            sample_X = f['X'][:1000]
            self.dataset_noise_std = np.std(sample_X)

        self.local_indices = np.arange(len(self.indices))
        np.random.shuffle(self.local_indices)
        self.total_len = len(self.indices)

    def __len__(self):
        return math.ceil(self.total_len / self.batch_size)

    def apply_fading_and_noise(self, X_source, M):
        """
        模拟 SIMO (单发多收) 信道
        X_source: (Batch, 1024, 2) - 原始发射信号
        M: 天线数
        """
        batch_size = X_source.shape[0]
        # 转为复数便于计算
        s = X_source[..., 0] + 1j * X_source[..., 1] # (Batch, 1024)
        
        # 1. 生成信道系数 h (Batch, M, 1)
        if self.fading == 'rayleigh':
            # 瑞利衰落: 实部虚部 ~ N(0, 1/sqrt(2))
            h = (np.random.normal(0, 1, (batch_size, M, 1)) + 
                 1j * np.random.normal(0, 1, (batch_size, M, 1))) / np.sqrt(2)
                 
        elif self.fading == 'rician':
            # 莱斯衰落 (K=10): 有视距分量
            k_factor = 10.0
            mu = np.sqrt(k_factor / (k_factor + 1))
            sigma = np.sqrt(1 / (2 * (k_factor + 1)))
            h_los = mu
            h_scat = sigma * (np.random.normal(0, 1, (batch_size, M, 1)) + 
                              1j * np.random.normal(0, 1, (batch_size, M, 1)))
            h = h_los + h_scat
            
        else: # awgn only (h=1)
            h = np.ones((batch_size, M, 1), dtype=np.complex64)
            
        # 2. 信号通过信道: y = h * s
        # (Batch, M, 1) * (Batch, 1, 1024) -> (Batch, M, 1024)
        s_expanded = np.expand_dims(s, 1) 
        y_clean = h * s_expanded 
        
        # 3. 生成独立噪声 (每个天线噪声不同)
        # 使用数据集原本的 noise level 作为基准
        n = (np.random.normal(0, self.dataset_noise_std, (batch_size, M, 1024)) + 
             1j * np.random.normal(0, self.dataset_noise_std, (batch_size, M, 1024)))
        
        y_noisy = y_clean + n
        
        # 转回实数 (Batch, M, 1024, 2)
        y_out = np.stack([np.real(y_noisy), np.imag(y_noisy)], axis=-1)
        return y_out.astype(np.float32)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.total_len)
        
        # 获取当前 batch 在 shuffled 列表中的位置
        batch_idx_in_local = self.local_indices[start:end]
        
        # 获取对应的真实文件索引
        batch_file_indices = self.indices[batch_idx_in_local]
        
        # 【关键修复】h5py 要求索引必须排序
        sorted_indices = np.sort(batch_file_indices)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            X_batch = f['X'][sorted_indices]
            
        # 为了避免因为排序导致的样本偏差，读取后再打乱一次顺序
        np.random.shuffle(X_batch)
        
        current_bs = X_batch.shape[0]
        
        if self.mode == 'binary':
            # 构造标签: 一半信号，一半噪声
            Y_batch = np.zeros((current_bs, 2), dtype=np.float32)
            sig_len = current_bs // 2
            
            # 1. 信号部分 (Label=[0, 1])
            Y_batch[:sig_len, 1] = 1.0 
            
            # 使用 batch 的前半部分作为信号源
            X_sig_source = X_batch[:sig_len]
            # 应用衰落生成多天线信号
            X_sig_final = self.apply_fading_and_noise(X_sig_source, self.num_nodes)
            
            # 2. 噪声部分 (Label=[1, 0])
            Y_batch[sig_len:, 0] = 1.0
            # 噪声不需要信道，直接生成 M 路独立噪声
            X_noise_final = np.random.normal(0, self.dataset_noise_std, 
                                           (current_bs - sig_len, self.num_nodes, 1024, 2)).astype(np.float32)
            
            X_final = np.concatenate([X_sig_final, X_noise_final], axis=0)
        else:
            # 非二分类模式直接全部应用衰落
            X_final = self.apply_fading_and_noise(X_batch, self.num_nodes)
            Y_batch = None 

        # --- 构图准备 ---
        # 模型输入期望: [X, A]
        # X shape: (Batch, M, 2048) -> 展平 IQ 维
        X_reshaped = X_final.reshape(current_bs, self.num_nodes, -1)
        
        # 计算空间相关性 (余弦相似度)
        norm = np.linalg.norm(X_reshaped, axis=2, keepdims=True) + 1e-8
        X_norm = X_reshaped / norm
        sim_matrix = np.matmul(X_norm, np.transpose(X_norm, (0, 2, 1))) # (B, M, M)
        
        # 构图 (绝对值相似度)
        A_batch = np.abs(sim_matrix)
        
        # 对称归一化
        D = np.sum(A_batch, axis=-1, keepdims=True)
        D_inv_sqrt = np.power(D + 1e-6, -0.5)
        A_norm = D_inv_sqrt * A_batch * np.transpose(D_inv_sqrt, (0, 2, 1))
        
        return [X_reshaped, A_norm.astype(np.float32)], Y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.local_indices)

def get_generators(hdf5_path, batch_size=32, num_nodes=2, split_ratio=0.8, max_samples=None, fading='rayleigh'):
    with h5py.File(hdf5_path, 'r') as f:
        total = f['X'].shape[0]
    if max_samples: total = min(total, max_samples)
    
    indices = np.arange(total)
    np.random.shuffle(indices)
    split = int(total * split_ratio)
    
    # 注意: num_nodes 传给 Sequence 作为 num_antennas
    train_gen = RadioMLSequence(hdf5_path, batch_size, indices[:split], num_antennas=num_nodes, mode='binary', fading=fading)
    val_gen = RadioMLSequence(hdf5_path, batch_size, indices[split:], num_antennas=num_nodes, mode='binary', fading=fading)
    
    return train_gen, val_gen, 2, train_gen.feature_dim