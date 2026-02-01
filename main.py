import os
import argparse
import tensorflow as tf
from dataset import get_generators 
from model import GCN_CSS, CNN_GCN_CSS, MLP_CSS
from training import train_model

# 显存设置
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to .hdf5 dataset')
    parser.add_argument('--model_type', type=str, default='cnn_gcn', choices=['gcn', 'cnn_gcn', 'mlp'])
    parser.add_argument('--nodes', type=int, default=2, help='Number of Antennas (M)')
    parser.add_argument('--fading', type=str, default='rayleigh', choices=['rayleigh', 'rician', 'awgn'], help='Channel Type')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32) 
    
    args = parser.parse_args()
    
    print(f"🚀 实验配置: M={args.nodes} (Antennas), Fading={args.fading}, Model={args.model_type}")
    
    # 1. 获取生成器
    train_gen, val_gen, num_classes, num_features = get_generators(
        hdf5_path=args.path,
        batch_size=args.batch_size,
        num_nodes=args.nodes, # 此时 nodes = antennas
        fading=args.fading
    )
    
    # 2. 构建模型
    if args.model_type == 'cnn_gcn':
        model = CNN_GCN_CSS(num_classes=num_classes, num_nodes=args.nodes)
        save_name = f'best_cnngcn_M{args.nodes}_{args.fading}.h5'
    elif args.model_type == 'gcn':
        model = GCN_CSS(num_classes=num_classes, num_nodes=args.nodes)
        save_name = f'best_gcn_M{args.nodes}_{args.fading}.h5'
    else:
        model = MLP_CSS(num_classes=num_classes, num_nodes=args.nodes)
        save_name = 'best_mlp.h5'

    # Build 模型 (输入: Batch, M, 2048)
    # 注意: num_features 在 dataset.py 里被强制设为 2048 (1024*2)
    model.build([(None, args.nodes, num_features), (None, args.nodes, args.nodes)])
    model.summary()
    
    # 3. 训练
    train_model(model, train_gen, val_gen, epochs=args.epochs, save_path=save_name)

if __name__ == "__main__":
    main()