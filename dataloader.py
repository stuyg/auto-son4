import tensorflow as tf
from dataset import compute_adjacency_matrix

def get_tf_dataset(X, Y, batch_size=32, shuffle=True):
    """
    构建 tf.data.Dataset，输出为 ((X, A), Y)
    """
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=1024)
    
    ds = ds.batch(batch_size)
    
    # 使用 map 函数在 GPU/CPU 上动态生成邻接矩阵 A
    def map_func(x_batch, y_batch):
        # x_batch: (Batch, Nodes, Feats)
        # 动态计算邻接矩阵 A: (Batch, Nodes, Nodes)
        A_batch = compute_adjacency_matrix(x_batch)
        return (x_batch, A_batch), y_batch

    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds