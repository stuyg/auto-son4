import tensorflow as tf
import os
import tensorflow.keras.backend as K

# ==========================================
# Focal Loss (保持不变)
# ==========================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        return K.sum(weight * cross_entropy, axis=-1)
    return focal_loss_fixed

# ==========================================
# 训练流程 (新增文件清理逻辑)
# ==========================================
def train_model(model, train_ds, val_ds, epochs=10, lr=0.001, save_path='best_model.h5'):
    # 【新增】: 如果旧权重文件存在，先删除，防止 h5py 写入冲突
    if os.path.exists(save_path):
        print(f"⚠️ 检测到旧权重文件 {save_path}，正在删除以避免冲突...")
        try:
            os.remove(save_path)
            print("✅ 旧文件已清除。")
        except OSError as e:
            print(f"❌ 无法删除旧文件: {e}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # 使用 Focal Loss
    loss_fn = categorical_focal_loss(gamma=2.0, alpha=0.25)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Checkpoint 回调
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path, 
        monitor='val_accuracy', 
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    print(f"🚀 开始训练 (M={model.layers[-1].units if hasattr(model, 'layers') and len(model.layers)>0 else '?'}, Loss=Focal)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )
    
    return history