import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# ==========================================
# 1. 图卷积层 (GraphConv)
# ==========================================
class GraphConv(layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.bn = layers.BatchNormalization()

    def build(self, input_shape):
        # input_shape: [(batch, nodes, feats), (batch, nodes, nodes)]
        feature_shape = input_shape[0] 
        input_dim = feature_shape[-1]
        
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        features, adjacency = inputs
        # features: (Batch, Nodes, InputDim)
        # adjacency: (Batch, Nodes, Nodes)
        
        # 1. 特征变换 (XW)
        support = tf.matmul(features, self.w)
        
        # 2. 邻居聚合 (AXW)
        output = tf.matmul(adjacency, support) + self.b
        
        # 3. BN + Activation
        output = self.bn(output)
        
        if self.activation is not None:
            output = self.activation(output)
        return output

# ==========================================
# 2. GCN 模型 (ResGCN 优化版)
# ==========================================
class GCN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(GCN_CSS, self).__init__()
        
        # --- 优化策略：残差连接 (ResNet style) ---
        # 为了使用残差连接 (x = f(x) + x)，输入输出维度必须一致。
        # 因此我们先用一个 Dense 层将输入投影到高维 (hidden_dim)。
        
        self.hidden_dim = 128  # 统一隐藏层维度
        
        # 1. 输入投影层 (Input Projection)
        self.input_proj = layers.Dense(self.hidden_dim, activation=None, name="input_proj")
        self.bn_proj = layers.BatchNormalization()
        
        # 2. GCN 块 (带残差)
        self.gcn1 = GraphConv(self.hidden_dim, activation='relu') 
        self.gcn2 = GraphConv(self.hidden_dim, activation='relu') 
        self.gcn3 = GraphConv(self.hidden_dim, activation='relu') 
        
        # 3. 分类头
        self.pooling = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        # x shape: (Batch, Nodes, Feats)
        # a shape: (Batch, Nodes, Nodes)
        
        # --- 投影到隐藏空间 ---
        x = self.input_proj(x)
        x = self.bn_proj(x)
        x = tf.nn.relu(x)
        
        # --- ResGCN Block 1 ---
        residual = x
        x = self.gcn1([x, a])
        x = x + residual  # 残差相加
        
        # --- ResGCN Block 2 ---
        residual = x
        x = self.gcn2([x, a])
        x = x + residual
        
        # --- ResGCN Block 3 ---
        residual = x
        x = self.gcn3([x, a])
        x = x + residual
        
        # --- Classification ---
        x = self.pooling(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

# ==========================================
# 3. CNN 模型 (保持不变)
# ==========================================
class CNN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(CNN_CSS, self).__init__()
        self.conv1 = layers.Conv1D(64, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling1D(2)
        
        self.conv2 = layers.Conv1D(128, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling1D(2)
        
        self.conv3 = layers.Conv1D(128, 3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        
        self.global_pool = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, _ = inputs 
        # 假设输入展平了，重塑回 (Batch, 1024, 2)
        # 注意：如果输入本身就是 (Batch, Nodes, Feats)，需要根据实际 Feat 维度调整
        # 这里为了兼容旧权重逻辑，假设输入总维度是 2048
        if len(x.shape) == 3:
            # (Batch, 32, 64) -> (Batch, 2048) -> (Batch, 1024, 2)
            batch = tf.shape(x)[0]
            x = tf.reshape(x, [batch, 1024, 2])
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = self.global_pool(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

# ==========================================
# 4. MLP 模型 (保持不变)
# ==========================================
class MLP_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(MLP_CSS, self).__init__()
        self.flatten = layers.Flatten()
        
        self.dense1 = layers.Dense(1024, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        
        self.dense2 = layers.Dense(512, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)
        
        self.fc_out = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, _ = inputs
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        return self.fc_out(x)
    
class GAT_CSS(Model):
    def __init__(self, num_classes, num_nodes, embed_dim=128, num_heads=4, ff_dim=128):
        super(GAT_CSS, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        
        # 1. 输入投影 + 位置编码
        self.input_proj = layers.Dense(embed_dim)
        # 位置编码：让模型知道节点的时间顺序 (Crucial for Time-Series)
        self.pos_emb = self.add_weight(
            name="pos_emb", shape=(1, num_nodes, embed_dim), 
            initializer="glorot_uniform", trainable=True
        )
        
        # 2. 注意力层 (GAT Layer 1)
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        # Feed Forward 1
        self.ffn1 = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.bn2 = layers.BatchNormalization()
        
        # 3. 注意力层 (GAT Layer 2)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.bn3 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        
        # Feed Forward 2
        self.ffn2 = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.bn4 = layers.BatchNormalization()
        
        # 4. 输出头
        self.pooling = layers.GlobalMaxPooling1D() # 或 GlobalAveragePooling1D
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout3 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # inputs: [X, A] 
        # 我们这里故意忽略 A (邻接矩阵)，因为它是基于噪声计算的，不准确。
        # 我们让 Attention 自己学习节点间的关系。
        x, _ = inputs 
        
        # (Batch, Nodes, Feats) -> (Batch, Nodes, Embed)
        x = self.input_proj(x)
        # 添加位置信息
        x = x + self.pos_emb 
        
        # --- Block 1 ---
        # Self-Attention: 自动寻找强相关的节点
        attn_output = self.att1(x, x)
        x = self.bn1(x + attn_output) # 残差 + BN
        x = self.dropout1(x)
        
        ffn_output = self.ffn1(x)
        x = self.bn2(x + ffn_output)  # 残差 + BN
        
        # --- Block 2 ---
        attn_output = self.att2(x, x)
        x = self.bn3(x + attn_output)
        x = self.dropout2(x)
        
        ffn_output = self.ffn2(x)
        x = self.bn4(x + ffn_output)
        
        # --- Output ---
        x = self.pooling(x)
        x = self.fc1(x)
        x = self.dropout3(x)
        return self.fc2(x)
    
class CNN_GCN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(CNN_GCN_CSS, self).__init__()
        self.num_nodes = num_nodes
        
        # --- 1. CNN 特征提取器 ---
        self.conv1 = layers.Conv1D(32, 3, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling1D(2) 
        
        self.conv2 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling1D(2)
        
        # --- 2. 特征投影 ---
        # 修正：确保输入维度是静态的，Dense 层才能正确 Build
        self.proj = layers.Dense(128, activation='relu')
        
        # --- 3. 动态图卷积 (GCN) ---
        self.gcn1 = GraphConv(128, activation='relu')
        self.gcn2 = GraphConv(128, activation='relu')
        
        # --- 4. 分类头 ---
        self.global_pool = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, _ = inputs 
        
        # 1. 还原为时序信号 
        # (Batch, 32, 64) -> (Batch, 1024, 2)
        # 使用动态 batch_size
        batch_size = tf.shape(x)[0]
        x_signal = tf.reshape(x, [batch_size, 1024, 2])
        
        # 2. CNN 提取特征
        h = self.conv1(x_signal) # -> (Batch, 1024, 32)
        h = self.bn1(h)
        h = self.pool1(h)        # -> (Batch, 512, 32)
        
        h = self.conv2(h)        # -> (Batch, 512, 64)
        h = self.bn2(h)
        h = self.pool2(h)        # -> (Batch, 256, 64)
        
        # 3. 重塑为图节点
        # 【关键修复】显式计算特征维度，而不是用 -1
        # 总特征数 = Time(256) * Channels(64) = 16384
        # 节点数 = self.num_nodes (通常是 32)
        # 每个节点的特征维度 = 16384 / 32 = 512
        total_feats = 256 * 64
        feats_per_node = total_feats // self.num_nodes
        
        # 使用显式维度进行 reshape，确保 Dense 层能获取 input_shape[-1]
        h_nodes = tf.reshape(h, [batch_size, self.num_nodes, feats_per_node])
        
        h_nodes = self.proj(h_nodes) # -> (Batch, 32, 128)
        
        # 4. 动态构建图 (Dynamic Graph Construction)
        att = tf.matmul(h_nodes, h_nodes, transpose_b=True)
        att = att / tf.sqrt(tf.cast(128, tf.float32))
        A_dynamic = tf.nn.softmax(att, axis=-1)
        
        # 5. GCN 处理
        x_gcn = self.gcn1([h_nodes, A_dynamic])
        x_gcn = x_gcn + h_nodes 
        
        x_out = self.gcn2([x_gcn, A_dynamic])
        x_out = x_out + x_gcn
        
        # 6. 分类
        out = self.global_pool(x_out)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)