# -*- coding: utf-8 -*-
"""Agri_project_local.py

Modified for local execution with batch training support.
"""

import os
import time
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- 0. GPU Verification ---

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {len(gpus)} device(s) found.")
        for gpu in gpus:
            print(f"   Name: {gpu.name}")
        # Optionally enable memory growth to avoid allocating all VRAM at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("❌ No GPU detected. TensorFlow will use the CPU.")

# --- 1. Configuration ---

# Use the current directory as the base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_DIR = os.path.join(BASE_DIR, 'crop leaves')
DATA_ROOT = os.path.join(BASE_DIR, 'data')
OUTPUT_ROOT = os.path.join(BASE_DIR, 'output')

IMG_SIZE = 224
TARGET_SAMPLES_PER_CLASS = 2500
BATCH_SIZE = 32

# Ensure root directories exist
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- 2. Shared Model Initialization ---

print("Loading VGG19 model for feature extraction...")
base_model = VGG19(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))

gap_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=gap_layer)
feature_extractor.trainable = False

print(f"VGG19 model loaded. Output feature vector shape: {feature_extractor.output_shape}")

# Augmentation Setup
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# --- 3. Helper Functions ---

def unzip_data(zip_path, extract_to):
    if not os.path.exists(extract_to) or not os.listdir(extract_to):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Data already extracted at {extract_to}")

def process_plant_dataset(plant_dir, output_dir, plant_name):
    """
    Loads, augments, and extracts features for a single plant.
    """
    start_time = time.time()
    print(f"\n--- Feature Extraction: {plant_name} ---")

    # Find class folders
    search_dir = plant_dir
    contents = os.listdir(search_dir)
    if len(contents) == 1 and os.path.isdir(os.path.join(search_dir, contents[0])):
        search_dir = os.path.join(search_dir, contents[0])

    class_names = [d for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d))]
    print(f"Found {len(class_names)} classes: {class_names}")

    if not class_names:
        print(f"Warning: No class folders found in {search_dir}")
        return None, None

    all_features = []
    all_labels = []

    for i, class_name in enumerate(class_names):
        print(f"Processing class ({i+1}/{len(class_names)}): {class_name}")
        
        temp_generator = datagen.flow_from_directory(
            search_dir,
            classes=[class_name],
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode=None,
            shuffle=True
        )

        class_features = []
        num_generated = 0

        while num_generated < TARGET_SAMPLES_PER_CLASS:
            try:
                img_batch = next(temp_generator)
            except StopIteration:
                break
            
            preprocessed_batch = preprocess_input(img_batch)
            feature_batch = feature_extractor.predict(preprocessed_batch, verbose=0)

            class_features.append(feature_batch)
            num_generated += len(img_batch)
            if num_generated % (BATCH_SIZE * 10) == 0:
                print(f"  Generated {num_generated}/{TARGET_SAMPLES_PER_CLASS} samples...", end='\r')

        if not class_features:
            continue

        class_features_np = np.vstack(class_features)
        all_features.append(class_features_np[:TARGET_SAMPLES_PER_CLASS])
        all_labels.append(np.full(len(all_features[-1]), i))

        print(f"  Final count: {len(all_features[-1])} samples for {class_name}.")

    if not all_features:
        return None, None

    # Combine, Shuffle, and Save
    final_features = np.vstack(all_features)
    final_labels = np.concatenate(all_labels)

    indices = np.arange(len(final_labels))
    np.random.shuffle(indices)
    final_features, final_labels = final_features[indices], final_labels[indices]

    os.makedirs(output_dir, exist_ok=True)
    feature_file = os.path.join(output_dir, f"{plant_name}_features.npy")
    label_file = os.path.join(output_dir, f"{plant_name}_labels.npy")

    np.save(feature_file, final_features)
    np.save(label_file, final_labels)

    print(f"Features and labels saved to {output_dir}")
    return feature_file, label_file

def train_svm(feature_file, label_file, plant_name, output_dir):
    print(f"--- Training SVM: {plant_name} ---")
    X = np.load(feature_file)
    y = np.load(label_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {(accuracy * 100):.2f}%")

    # Save
    model_save_path = os.path.join(output_dir, f'{plant_name}_model.pkl')
    joblib.dump(model, model_save_path)
    print(f"Model saved to: {model_save_path}")

# --- 4. Main Execution ---

if __name__ == "__main__":
    check_gpu()
    # Get all zip files from the 'crop leaves' directory
    if not os.path.exists(ZIP_DIR):
        print(f"Error: Directory '{ZIP_DIR}' not found.")
    else:
        zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith('.zip')]
        print(f"Found {len(zip_files)} plant datasets in {ZIP_DIR}")

        for zip_name in zip_files:
            plant_name = zip_name.replace('.zip', '')
            print(f"\n{'='*60}")
            print(f" STARTING: {plant_name}")
            print(f"{'='*60}")

            zip_path = os.path.join(ZIP_DIR, zip_name)
            extract_path = os.path.join(DATA_ROOT, plant_name)
            plant_output_dir = os.path.join(OUTPUT_ROOT, plant_name)

            try:
                # 1. Unzip
                unzip_data(zip_path, extract_path)

                # 2. Extract Features
                feat_file, lab_file = process_plant_dataset(extract_path, plant_output_dir, plant_name.lower())

                if feat_file and lab_file:
                    # 3. Train
                    train_svm(feat_file, lab_file, plant_name, plant_output_dir)
                    print(f"SUCCESS: Finished processing {plant_name}")
                else:
                    print(f"SKIPPING: No features extracted for {plant_name}")

            except Exception as e:
                print(f"ERROR processing {plant_name}: {e}")

            # Optional: clear session to free up memory (mostly for TF)
            # tf.keras.backend.clear_session()
            
        print(f"\n{'='*60}")
        print(" BATCH TRAINING COMPLETE!")
        print(f"{'='*60}")