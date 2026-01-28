# 修改文件: stuyg/auto-son2/.../training.py

import tensorflow as tf
import os

# 修改函数签名，增加 save_path 参数
def train_model(model, train_ds, val_ds, epochs=10, lr=0.001, save_path='best_model.h5'):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # 使用传入的 save_path
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

    print(f"🚀 开始训练 (权重将保存至: {save_path})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )
    
    return history