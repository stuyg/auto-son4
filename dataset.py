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
        
        print(f"正在加载 {len(indices)} 条数据...")
        with h5py.File(self.hdf5_path, 'r') as f:
            self.feature_dim = f['X'].shape[1] * f['X'].shape[2] // self.num_nodes
            sorted_indices = np.sort(self.indices)
            self.X_data = f['X'][sorted_indices]
            self.Z_data = f['Z'][sorted_indices]
            
            if mode != 'binary':
                self.Y_data = f['Y'][sorted_indices]
            else:
                self.Y_data = None
                
            # 【核心修改】计算全局物理底噪
            # 我们随机采样一些数据来寻找最小 SNR 的能量水平
            sample_size = min(2000, len(self.indices))
            temp_Z = self.Z_data[:sample_size]
            temp_X = self.X_data[:sample_size]
            
            # 找到最小 SNR
            min_snr = np.min(temp_Z)
            noise_idx = np.where(temp_Z == min_snr)[0]
            
            if len(noise_idx) > 0:
                self.noise_std = np.std(temp_X[noise_idx])
            else:
                # 兜底：直接取所有样本中能量最小的那部分
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
        batch_idx = self.local_indices[start:end]
        
        X_batch = self.X_data[batch_idx].copy()
        Z_batch = self.Z_data[batch_idx].copy()

        if self.mode == 'binary':
            Y_new = np.zeros((current_batch_size, 2), dtype=np.float32)
            Y_new[:, 1] = 1.0 
            
            noise_count = current_batch_size // 2
            if noise_count > 0:
                # 【核心修改】使用固定的物理底噪生成 H0
                noise_data = np.random.normal(0, self.noise_std, size=(noise_count, 1024, 2))
                
                X_batch[-noise_count:] = noise_data
                Y_new[-noise_count:, 0] = 1.0
                Y_new[-noise_count:, 1] = 0.0
                Z_batch[-noise_count:] = -100

            Y_batch = Y_new
        else:
            Y_batch = self.Y_data[batch_idx]

        X_reshaped = X_batch.reshape(-1, self.num_nodes, self.feature_dim)
        X_tensor = tf.convert_to_tensor(X_reshaped, dtype=tf.float32)
        
        diff = tf.expand_dims(X_tensor, 2) - tf.expand_dims(X_tensor, 1)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
        A_batch = tf.exp(-dist_sq / (self.sigma ** 2))
        D = tf.reduce_sum(A_batch, axis=-1, keepdims=True)
        A_batch_norm = A_batch / (D + 1e-6)

        return [X_tensor, A_batch_norm], Y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.local_indices)

# 记得保留 get_generators 函数，它需要调用上面的 RadioMLSequence
def get_generators(hdf5_path, batch_size=32, num_nodes=32, split_ratio=0.8, max_samples=None):
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = f['X'].shape[0]
    if max_samples: total_samples = min(total_samples, max_samples)
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    split_idx = int(total_samples * split_ratio)
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    train_gen = RadioMLSequence(hdf5_path, batch_size, train_indices, num_nodes, mode='binary')
    val_gen = RadioMLSequence(hdf5_path, batch_size, val_indices, num_nodes, mode='binary')
    return train_gen, val_gen, 2, train_gen.feature_dim