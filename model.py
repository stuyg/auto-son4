import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

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
        if self.activation is not None:
            output = self.activation(output)
        return output

# ==========================================
# GCN 模型 (支持多天线输入)
# ==========================================
class GCN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(GCN_CSS, self).__init__()
        # 直接处理特征，不经过 CNN
        self.input_proj = layers.Dense(128, activation='relu')
        self.gcn1 = GraphConv(128, activation='relu') 
        self.gcn2 = GraphConv(128, activation='relu') 
        self.pooling = layers.GlobalMaxPooling1D() # 空间最大池化
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.input_proj(x)
        x = self.gcn1([x, a])
        x = self.gcn2([x, a])
        x = self.pooling(x)
        return self.fc(x)

# ==========================================
# CNN-GCN 融合模型 (SOTA 架构修改版)
# 专为多天线设计: Shared CNN -> Spatial GCN
# ==========================================
class CNN_GCN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(CNN_GCN_CSS, self).__init__()
        self.num_nodes = num_nodes # 天线数 M
        
        # 1. 共享特征提取器 (Shared CNN)
        # 输入: (Batch * M, 1024, 2)
        self.cnn_backbone = tf.keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.GlobalMaxPooling1D() # 输出 (Batch * M, 64)
        ])
        
        # 2. 特征投影
        self.proj = layers.Dense(128, activation='relu')
        
        # 3. 空间图卷积 (融合不同天线的信息)
        self.gcn1 = GraphConv(128, activation='relu')
        self.gcn2 = GraphConv(128, activation='relu')
        
        # 4. 分类头
        self.global_pool = layers.GlobalMaxPooling1D() # 跨天线池化
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs 
        # x shape: (Batch, M, 2048) -> 来自 dataset.py 的 flat 输出
        # a shape: (Batch, M, M)
        
        batch_size = tf.shape(x)[0]
        
        # 1. 重塑为 (Batch * M, 1024, 2) 以便通过共享 CNN
        x_reshaped = tf.reshape(x, [-1, 1024, 2])
        
        # 2. 提取每根天线的特征
        feats = self.cnn_backbone(x_reshaped) # -> (Batch * M, 64)
        
        # 3. 还原为图结构 (Batch, M, 64)
        feats_graph = tf.reshape(feats, [batch_size, self.num_nodes, 64])
        feats_graph = self.proj(feats_graph) # -> (Batch, M, 128)
        
        # 4. GCN 融合
        x_gcn = self.gcn1([feats_graph, a])
        x_gcn = x_gcn + feats_graph # 残差
        
        x_out = self.gcn2([x_gcn, a])
        x_out = x_out + x_gcn # 残差
        
        # 5. 全局分类
        out = self.global_pool(x_out) # 融合所有天线结果
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

# 保留 MLP 和 CNN 以防需要对比 (接口不做大改，仅作为 Baseline)
class MLP_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(MLP_CSS, self).__init__()
        self.flatten = layers.Flatten()
        self.fc_out = layers.Dense(num_classes, activation='softmax')
    def call(self, inputs):
        x, _ = inputs
        x = self.flatten(x)
        return self.fc_out(x)