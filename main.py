import matplotlib.pyplot as plt
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from math import ceil


# ---------------- KONFIGURASI ----------------

# Sesuaikan path ke folder dataset Anda
DATASET_PATH = "asl_alphabet_train" 
# Direktori untuk menyimpan semua output
OUTPUT_DIR = "output"

# Parameter Model dan Training
IMG_SIZE = 64
BATCH_SIZE = 128 # Batch size bisa diperbesar untuk training lebih cepat jika VRAM cukup
EPOCHS = 50

# Membuat direktori output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- DATA PREPARATION ----------------

# Data Augmentation untuk training dan normalisasi untuk validasi
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split 20% data untuk validasi
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # Pastikan validation_split sama
)

print("Mempersiapkan data generator...")
# Generator untuk data training
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training', # Mengambil data training
    color_mode='rgb'
)

# Generator untuk data validasi
validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation', # Mengambil data validasi
    color_mode='rgb'
)

# Dapatkan jumlah kelas dan nama kelas
num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())
print(f"Ditemukan {train_generator.n} gambar training dan {validation_generator.n} gambar validasi dalam {num_classes} kelas.")

# Simpan nama kelas
class_names_path = os.path.join(OUTPUT_DIR, 'class_names.pkl')
with open(class_names_path, 'wb') as f:
    pickle.dump(class_names, f)
print(f"Nama kelas disimpan di {class_names_path}")


# ---------------- MODEL DEFINITION ----------------

def create_model(input_shape, num_classes):
    model = Sequential([
        # Definisikan input_shape langsung di layer pertama
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Buat model
input_shape = (IMG_SIZE, IMG_SIZE, 3)
model = create_model(input_shape, num_classes)

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Tampilkan summary model
model.summary()


# ---------------- MODEL TRAINING ----------------

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

print("\nMemulai training model...")
history = model.fit(
    train_generator,
    steps_per_epoch=ceil(train_generator.samples / BATCH_SIZE),
    validation_data=validation_generator,
    validation_steps=ceil(validation_generator.samples / BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("Training selesai.")

# ---------------- EVALUATION & SAVING ----------------

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
history_plot_path = os.path.join(OUTPUT_DIR, 'training_history.png')
plt.savefig(history_plot_path)
plt.show()
print(f"Plot training disimpan di {history_plot_path}")

# Simpan model
model_path = os.path.join(OUTPUT_DIR, 'sign_language_model.keras')
model.save(model_path)
print(f"Model disimpan di {model_path}")