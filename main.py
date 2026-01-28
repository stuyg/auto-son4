import os
import argparse
import tensorflow as tf

# ==========================================
# 1. 显存配置
# ==========================================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
# 【关键修复】这里必须包含 get_generators
from dataset import get_generators 
from model import GCN_CSS, CNN_CSS, MLP_CSS 
from training import train_model

def main():
    parser = argparse.ArgumentParser(description="GCN/CNN/MLP Spectrum Sensing")
    parser.add_argument('--path', type=str, required=True, help='Path to .hdf5 dataset')
    # 支持模型选择
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'cnn', 'mlp'], help='Choose model architecture')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--samples', type=int, default=None)
    
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
    
    # Build 模型
    # 注意：GCN 需要两个输入 [(Batch, Nodes, Feats), (Batch, Nodes, Nodes)]
    # CNN/MLP 虽然只用 Feats，但为了接口统一，这里 Build 形状保持一致即可
    model.build([(None, args.nodes, num_features), (None, args.nodes, args.nodes)])
    model.summary()
    
    # 开始训练
    # 注意：请确保你的 training.py 已经按照上一步修改，支持 save_path 参数
    train_model(model, train_gen, val_gen, epochs=args.epochs, lr=args.lr, save_path=save_name)

if __name__ == "__main__":
    main()