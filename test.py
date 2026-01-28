import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCNConv, GlobalSumPool
from scipy.spatial.distance import cdist

# ==========================================
# 1. 内存安全的数据生成器
# ==========================================
class RadioGraphGenerator:
    def __init__(self, h5_path, batch_size=32, num_users=15, antennas_per_user=15):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_users = num_users
        self.antennas_per_user = antennas_per_user
        self.num_nodes = num_users * antennas_per_user  # 225
        
        # 预先构建拓扑结构 (邻接矩阵 A) [cite: 201-202]
        self.A_matrix = self._build_topology()
        
        # 获取 QPSK 索引 (只读标签，不占内存)
        print("正在扫描数据集索引 (这可能需要几秒钟)...")
        self.h1_indices = self._get_qpsk_indices()
        print(f"找到 QPSK (H1) 样本数: {len(self.h1_indices)}")
        
    def _build_topology(self, area_size=100, rho=10.0):
        """构建 15x15 的空间位置并计算邻接矩阵 A"""
        positions = []
        user_centers = np.random.rand(self.num_users, 2) * area_size
        for center in user_centers:
            # 模拟用户周围的天线簇
            antennas = center + np.random.randn(self.antennas_per_user, 2) * 0.5
            positions.extend(antennas)
        positions = np.array(positions)
        # 计算距离并应用 RBF 核 [cite: 202]
        dist_matrix = cdist(positions, positions, metric='euclidean')
        A = np.exp(-(dist_matrix**2) / (rho**2))
        return A

    def _get_qpsk_indices(self):
        """只读取标签 Y，定位 QPSK 数据的位置"""
        # RadioML 2018.01A 类别顺序
        classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                   '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                   '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                   'FM', 'GMSK', 'OQPSK']
        target_idx = classes.index('QPSK')
        
        with h5py.File(self.h5_path, 'r') as f:
            # 这里的 argmax 可能也会耗内存，如果只有 8G 内存，建议分块读取
            # 这里假设机器能存下标签数组 (约几百MB)
            y = np.argmax(f['Y'][:], axis=1)
            indices = np.where(y == target_idx)[0]
        return indices

    def _calculate_energy(self, iq_samples):
        """能量检测: (Batch, 1024, 2) -> (Batch, 1) [cite: 152]"""
        return np.mean(np.sum(iq_samples**2, axis=2), axis=1, keepdims=True)

    def generator(self):
        """
        核心生成器：无限循环，每次只读 1 个 batch 的数据
        """
        with h5py.File(self.h5_path, 'r') as f:
            X_ds = f['X'] # 获取句柄，不加载数据
            
            while True:
                # 准备容器
                batch_X = []
                batch_A = []
                batch_y = []
                
                for _ in range(self.batch_size):
                    # 随机决定是 H1 (有PU) 还是 H0 (噪声)
                    label = 1 if np.random.rand() > 0.5 else 0
                    
                    if label == 1:
                        # --- H1: 读硬盘 ---
                        # 随机抽 225 个 QPSK 样本的索引
                        sample_indices = np.random.choice(self.h1_indices, self.num_nodes, replace=False)
                        sample_indices.sort() # h5py 要求索引必须排序
                        
                        # 【关键】只读取这 225 个样本
                        raw_iq = X_ds[sample_indices]
                        features = self._calculate_energy(raw_iq)
                        
                    else:
                        # --- H0: 生成噪声 ---
                        # 噪声方差根据数据集底噪调整，假设为 0.01
                        noise = np.random.normal(0, np.sqrt(0.01/2), (self.num_nodes, 1024, 2))
                        features = self._calculate_energy(noise)
                    
                    batch_X.append(features)
                    batch_A.append(self.A_matrix)
                    batch_y.append(label)
                
                # 转换为 Numpy 格式
                # Keras 多输入格式: [X_input, A_input], label
                yield [np.array(batch_X), np.array(batch_A)], np.array(batch_y)

# ==========================================
# 2. 构建 GCN 模型 (基于论文 Table II)
# ==========================================
def build_model(num_nodes=225):
    # 输入层: X (特征) 和 A (邻接矩阵)
    X_in = Input(shape=(num_nodes, 1), name='X_input')
    A_in = Input(shape=(num_nodes, num_nodes), name='A_input')

    # 图卷积层
    gc1 = GCNConv(32, activation='relu')([X_in, A_in])
    gc2 = GCNConv(64, activation='relu')([gc1, A_in])
    gc3 = GCNConv(128, activation='relu')([gc2, A_in])

    # 图池化层
    pool = GlobalSumPool()(gc3)

    # 全连接层
    fc1 = Dense(64, activation='relu')(pool)
    output = Dense(2, activation='softmax')(fc1)

    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # 替换你的 h5 文件路径
    H5_FILE_PATH = 'GOLD_XYZ_OSC.0001_1024.hdf5' 
    
    # 1. 实例化生成器
    # 这一步只读取轻量级的索引，不会爆内存
    data_gen = RadioGraphGenerator(H5_FILE_PATH, batch_size=32)
    
    # 2. 构建模型
    model = build_model(num_nodes=225)
    model.summary()
    
    # 3. 开始训练
    # steps_per_epoch: 每一个 epoch 训练多少个 batch
    # 这里的生成器是无限数据的，所以必须指定 steps_per_epoch
    try:
        print("开始训练...")
        model.fit(
            data_gen.generator(), 
            steps_per_epoch=50,  # 测试用，正式训练可设为 1000+
            epochs=5
        )
        print("训练完成！")
    except KeyboardInterrupt:
        print("训练手动停止")