import tensorflow as tf
import numpy as np
import cv2
import os
from parse_veri import parse_veri_labels

# 1. SETUP DATA
VERI_BASE_PATH = r"C:\VeRi\VeRi"
training_samples = parse_veri_labels(os.path.join(VERI_BASE_PATH, "train_label.xml"), 
                                     os.path.join(VERI_BASE_PATH, "image_train"))

# ID Mapping
unique_ids = sorted(list(set([s['vehicle_id'] for s in training_samples])))
id_to_idx = {vid: i for i, vid in enumerate(unique_ids)}
num_classes = len(unique_ids)

# 2. BUILD THE ARCHITECTURE (Matches mars.pb logic)
def build_small128(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 64, 3), name="images"),
        
        # Convolutional Block 1
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        # Convolutional Block 2
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Flatten(),
        
        # THE CRITICAL LAYER: 128-dimensional feature vector
        # This name 'features' must match what your freeze_model.py looks for
        tf.keras.layers.Dense(128, activation='relu', name='features'),
        
        tf.keras.layers.Dropout(0.5),
        # Softmax layer for 576 vehicle classes
        tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')
    ])
    return model

# 3. GENERATOR & TRAINING
def data_generator(samples, batch_size=32):
    while True:
        np.random.shuffle(samples)
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            X = [cv2.resize(cv2.imread(s['image_path']), (64, 128)) / 255.0 for s in batch]
            y = [id_to_idx[s['vehicle_id']] for s in batch]
            yield np.array(X), np.array(y)

model = build_small128(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"Starting training for {num_classes} vehicles...")
model.fit(data_generator(training_samples), 
          steps_per_epoch=len(training_samples)//32, 
          epochs=20) # 20 epochs is a good baseline for Re-ID

# 4. SAVE CHECKPOINT
# This saves the weights so freeze_model.py can read them
model.save_weights("veri_vehicle_only.ckpt")
print("✅ Training complete. Weights saved as veri_vehicle_only.ckpt")