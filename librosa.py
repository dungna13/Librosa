!pip install librosa scikit-learn tensorflow keras

import os
import matplotlib.pyplot as plt
import csv
import re  # Thư viện để xử lý chuỗi
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Đường dẫn đến các thư mục chứa dữ liệu
train_abnormal_folder = '/kaggle/input/trainning/model/train/abnormal'
train_normal_folder = '/kaggle/input/trainning/model/train/normal'

# Nạp dữ liệu từ các file .npy
def load_data_from_folder(folder, target_size=(224, 224)):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder, filename)
            array = np.load(file_path)
            # Thay đổi kích thước
            array_resized = cv2.resize(array, target_size)  # Thay đổi kích thước thành (224, 224)
            data.append(array_resized)
    return data

# Nạp dữ liệu train và test
train_abnormal_data = load_data_from_folder(train_abnormal_folder)
train_normal_data = load_data_from_folder(train_normal_folder)


# Chuyển đổi dữ liệu và gán nhãn (1 cho abnormal, 0 cho normal)
X_train = np.concatenate([train_abnormal_data, train_normal_data], axis=0)
y_train = np.concatenate([np.ones(len(train_abnormal_data)), np.zeros(len(train_normal_data))], axis=0)


# Thay đổi kích thước đầu vào cho phù hợp với MobileNet
if X_train.ndim == 3:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

# Chuyển đổi dữ liệu sang 3 kênh (RGB)
if X_train.shape[-1] == 1:
    X_train = np.repeat(X_train, 3, axis=-1)

# Khởi tạo mô hình MobileNet
mobilenet = ResNet50(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], 3))

# Thay đổi trainable layers
for layer in mobilenet.layers[:-24]:  # Chỉ cho phép một số lớp đầu tiên huấn luyện
    layer.trainable = False

# Xây dựng mô hình Sequential
model = Sequential()
model.add(mobilenet)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate =1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Tạo đối tượng tăng cường dữ liệu
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=50,
          validation_data=(X_train, y_train),
          callbacks=[early_stopping])

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_train, y_train)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


# Lưu mô hình
model.save('/kaggle/working/trained_model.h5')

