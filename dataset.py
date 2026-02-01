import h5py
import numpy as np
import tensorflow as tf
import math

class RadioMLSequence(tf.keras.utils.Sequence):
    def __init__(self, hdf5_path, batch_size, indices, num_nodes=32, sigma=1.0, mode='binary'):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.indices = indices
        self.num_nodes = num_nodes
        self.sigma = sigma
        self.mode = mode
        self.num_classes = 2 if mode == 'binary' else 24
        
        print(f"正在初始化生成器 (总样本数: {len(indices)})...")
        
        # 初始化读取少量数据计算底噪
        with h5py.File(self.hdf5_path, 'r') as f:
            self.feature_dim = f['X'].shape[1] * f['X'].shape[2] // self.num_nodes
            
            sample_size = min(2000, len(self.indices))
            sample_indices = np.sort(self.indices[:sample_size])
            
            temp_Z = f['Z'][sample_indices]
            temp_X = f['X'][sample_indices]
            
            min_snr = np.min(temp_Z)
            noise_idx = np.where(temp_Z == min_snr)[0]
            
            if len(noise_idx) > 0:
                self.noise_std = np.std(temp_X[noise_idx])
            else:
                powers = np.mean(np.var(temp_X, axis=1), axis=1)
                self.noise_std = np.sqrt(np.min(powers))
                
            print(f"✅ 训练集底噪基准 Std: {self.noise_std:.6f} (基于 {min_snr}dB 样本)")

        self.local_indices = np.arange(len(self.indices))
        np.random.shuffle(self.local_indices)
        self.total_len = len(self.indices)

    def __len__(self):
        return math.ceil(self.total_len / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.total_len)
        current_batch_size = end - start
        
        batch_idx_in_indices = self.local_indices[start:end]
        file_indices = self.indices[batch_idx_in_indices]
        sorted_file_indices = np.sort(file_indices)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            X_batch = f['X'][sorted_file_indices]
            Z_batch = f['Z'][sorted_file_indices]
            if self.mode != 'binary':
                Y_original = f['Y'][sorted_file_indices]
            else:
                Y_original = None

        X_batch = X_batch.astype(np.float32)

        if self.mode == 'binary':
            Y_new = np.zeros((current_batch_size, 2), dtype=np.float32)
            Y_new[:, 1] = 1.0 
            
            noise_count = current_batch_size // 2
            if noise_count > 0:
                noise_data = np.random.normal(0, self.noise_std, size=(noise_count, 1024, 2)).astype(np.float32)
                X_batch[-noise_count:] = noise_data
                Y_new[-noise_count:, 0] = 1.0
                Y_new[-noise_count:, 1] = 0.0
                Z_batch[-noise_count:] = -100
            Y_batch = Y_new
        else:
            Y_batch = Y_original

        # ==========================================
        # 优化点：对称归一化 (Symmetric Normalization)
        # ==========================================
        X_reshaped = X_batch.reshape(-1, self.num_nodes, self.feature_dim)
        
        # 计算欧氏距离
        diff = np.expand_dims(X_reshaped, 2) - np.expand_dims(X_reshaped, 1)
        dist_sq = np.sum(np.square(diff), axis=-1)
        
        # 建议：减小 sigma 以稀疏化图 (例如 sigma=0.5)
        # 如果你想在这里硬编码，可以改为 self.sigma ** 2 (假设外部传入了0.5)
        A_batch = np.exp(-dist_sq / (self.sigma ** 2))
        
        # 对称归一化: D^-0.5 * A * D^-0.5
        D = np.sum(A_batch, axis=-1, keepdims=True)
        # 防止除零
        D_inv_sqrt = np.power(D + 1e-6, -0.5) 
        
        # 利用广播: (N, 1) * (N, N) * (1, N)
        # 注意 numpy 广播规则，D_inv_sqrt 需要转置乘在右边
        # formula: D_inv_sqrt * A * Transpose(D_inv_sqrt)
        A_batch_norm = D_inv_sqrt * A_batch * np.transpose(D_inv_sqrt, (0, 2, 1))

        return [X_reshaped, A_batch_norm.astype(np.float32)], Y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.local_indices)

def get_generators(hdf5_path, batch_size=32, num_nodes=32, split_ratio=0.8, max_samples=None):
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = f['X'].shape[0]
    if max_samples: total_samples = min(total_samples, max_samples)
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    split_idx = int(total_samples * split_ratio)
    
    # 修正点：参数名统一为 indices
    train_gen = RadioMLSequence(hdf5_path, batch_size, indices=all_indices[:split_idx], num_nodes=num_nodes, mode='binary')
    val_gen = RadioMLSequence(hdf5_path, batch_size, indices=all_indices[split_idx:], num_nodes=num_nodes, mode='binary')
    
    return train_gen, val_gen, 2, train_gen.feature_dim