import os
import argparse

# ==========================================
# 1. 显存配置 (务必放在 import tensorflow 之前)
# ==========================================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ [GPU] 已检测到 {len(gpus)} 个 GPU，显存动态增长已开启。")
    except RuntimeError as e:
        print(f"❌ 显存设置失败: {e}")
else:
    print("⚠️ 未检测到 GPU，将使用 CPU 运行。")

# ==========================================
# 2. 导入自定义模块
# ==========================================
from dataset import get_generators 
from model import GCN_CSS, CNN_CSS, MLP_CSS, GAT_CSS,CNN_GCN_CSS
from training import train_model

def main():
    parser = argparse.ArgumentParser(description="GCN/CNN/MLP Spectrum Sensing")
    parser.add_argument('--path', type=str, required=True, help='Path to .hdf5 dataset')
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'cnn', 'mlp', 'gat','cnn_gcn'], help='Choose model architecture')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--samples', type=int, default=None)
    
    # 【新增】支持断点续训的参数
    parser.add_argument('--resume', action='store_true', help='Resume training from the best checkpoint if available')
    
    args = parser.parse_args()
    
    print(f"🚀 正在准备数据生成器 (Nodes={args.nodes})...")
    
    # 获取生成器
    train_gen, val_gen, num_classes, num_features = get_generators(
        hdf5_path=args.path,
        batch_size=args.batch_size,
        num_nodes=args.nodes,
        split_ratio=0.8,
        max_samples=args.samples
    )
    
    print(f"生成器准备完毕。分类数: {num_classes}, 节点特征维数: {num_features}")
    
    # 根据参数选择模型和保存路径
    if args.model_type == 'gcn':
        print("构建 GCN 模型...")
        model = GCN_CSS(num_classes=num_classes, num_nodes=args.nodes)
        save_name = 'best_gcn_model.h5'
    elif args.model_type == 'cnn':
        print("构建 CNN 模型...")
        model = CNN_CSS(num_classes=num_classes, num_nodes=args.nodes)
        save_name = 'best_cnn_model.h5'
    elif args.model_type == 'mlp':
        print("构建 MLP 模型...")
        model = MLP_CSS(num_classes=num_classes, num_nodes=args.nodes)
        save_name = 'best_mlp_model.h5'
    elif args.model_type == 'gat':
        print("构建 GAT (Graph Transformer) 模型...")
        model = GAT_CSS(num_classes=num_classes, num_nodes=args.nodes)
        save_name = 'best_gat_model.h5'
    elif args.model_type == 'cnn_gcn':
        print("构建 CNN-GCN 融合模型 (SOTA)...")
        model = CNN_GCN_CSS(num_classes=num_classes, num_nodes=args.nodes)
        save_name = 'best_cnngcn_model.h5'
    
    # Build 模型
    # GCN 需要两个输入，CNN/MLP 为了接口统一也build成相同形状
    model.build([(None, args.nodes, num_features), (None, args.nodes, args.nodes)])
    model.summary()
    
    # 【新增】断点续训逻辑
    if args.resume:
        if os.path.exists(save_name):
            print(f"🔄 检测到断点续训请求，正在加载权重: {save_name}")
            try:
                model.load_weights(save_name)
                print("✅ 权重加载成功，将基于现有模型继续训练。")
            except Exception as e:
                print(f"❌ 权重加载失败: {e}，将重新开始训练。")
        else:
            print(f"⚠️ 未找到权重文件 {save_name}，无法续训，将重新开始训练。")
    
    # 开始训练
    train_model(model, train_gen, val_gen, epochs=args.epochs, lr=args.lr, save_path=save_name)

if __name__ == "__main__":
    main()