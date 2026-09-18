import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure PyTorch backend for Keras
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def train_and_evaluate_cnn():
    dataset_path = os.path.join("processed_data", "ecg_dataset.npz")
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at '{dataset_path}'. Run preprocess.py first.")

    print("=" * 75)
    print("             1D CNN ECG RHYTHM CLASSIFICATION TRAINING             ")
    print("=" * 75)

    # 1. Load existing leakage-free dataset
    data = np.load(dataset_path)
    X_train = data['X_train']  # Shape: (790, 3600)
    X_val = data['X_val']      # Shape: (210, 3600)
    y_train = data['y_train']
    y_val = data['y_val']
    class_names = [str(c) for c in data['class_names']]
    num_classes = len(class_names)

    # Reshape for 1D CNN: (samples, timesteps, channels) -> (N, 3600, 1)
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_val_cnn = np.expand_dims(X_val, axis=-1)

    print(f"\n[1] Data Inspection (Grouped Split Maintained):")
    print(f" - Train Input Shape : {X_train_cnn.shape} (Labels: {y_train.shape})")
    print(f" - Val Input Shape   : {X_val_cnn.shape} (Labels: {y_val.shape})")
    print(f" - Total Classes     : {num_classes}")

    # 2. Handle class imbalance safely (clipped weights to avoid extreme bias towards train-only classes)
    print("\n[2] Computing Balanced Class Weights for Imbalance Handling...")
    unique_classes_train = np.unique(y_train)
    raw_weights = compute_class_weight('balanced', classes=unique_classes_train, y=y_train)
    # Clip extreme weights to range [0.5, 3.0] to prevent train-only rare classes from distorting logits
    clipped_weights = np.clip(raw_weights, 0.5, 3.0)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes_train, clipped_weights)}
    print(f"Clipped weights for {len(class_weight_dict)} active training classes (range: [0.5, 3.0]).")

    # 3. Build 1D CNN Architecture (Requested Structure)
    print("\n[3] Building 1D CNN Architecture...")
    # Conv1D -> BatchNorm -> ReLU -> MaxPooling
    # Conv1D -> BatchNorm -> ReLU -> MaxPooling
    # Conv1D -> BatchNorm -> ReLU -> GlobalAveragePooling
    # Dense -> Dropout -> Output layer
    model = models.Sequential([
        # Block 1
        layers.Input(shape=(3600, 1)),
        layers.Conv1D(filters=32, kernel_size=15, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),

        # Block 2
        layers.Conv1D(filters=64, kernel_size=11, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),

        # Block 3
        layers.Conv1D(filters=128, kernel_size=7, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling1D(),

        # Dense Classifier Block
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 4. Training Callbacks & Execution
    model_save_path = os.path.join(model_dir, "ecg_cnn.keras")
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    print("\n[4] Training 1D CNN Model (Max Epochs=40, Batch Size=32)...")
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=40,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    print(f"\nSaved best 1D CNN model to '{model_save_path}'")

    # 5. Plotting Training & Validation Loss & Accuracy
    print("\n[5] Generating Training / Validation Curves...")
    epochs_range = range(1, len(history.history['loss']) + 1)

    # Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history['loss'], 'b-o', label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('1D CNN - Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    loss_fig_path = os.path.join("processed_data", "cnn_loss_curve.png")
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history['accuracy'], 'b-o', label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], 'r-s', label='Validation Accuracy')
    plt.title('1D CNN - Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    acc_fig_path = os.path.join("processed_data", "cnn_accuracy_curve.png")
    plt.savefig(acc_fig_path, dpi=300)
    plt.close()

    print(f" - Saved Loss Curve to '{loss_fig_path}'")
    print(f" - Saved Accuracy Curve to '{acc_fig_path}'")

    # 6. Evaluation on Validation Set
    print("\n[6] Evaluating 1D CNN Model on Validation Set...")
    val_probs = model.predict(X_val_cnn)
    val_preds = np.argmax(val_probs, axis=1)

    acc = accuracy_score(y_val, val_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_val, val_preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_val, val_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 75)
    print("                    1D CNN VALIDATION METRICS                    ")
    print("=" * 75)
    print(f" Validation Accuracy : {acc * 100:.2f}%")
    print(f" Macro Precision    : {p_macro * 100:.2f}%")
    print(f" Macro Recall       : {r_macro * 100:.2f}%")
    print(f" Macro F1-Score     : {f1_macro * 100:.2f}%")
    print(f" Weighted F1-Score  : {f1_weighted * 100:.2f}%")
    print("=" * 75)

    # Per-Class Report
    val_unique_labels = sorted(list(set(y_val)))
    val_target_names = [class_names[cid] for cid in val_unique_labels]

    print("\n--- [A] PER-CLASS CLASSIFICATION REPORT ---")
    report = classification_report(
        y_val, val_preds,
        labels=val_unique_labels,
        target_names=val_target_names,
        zero_division=0
    )
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_val, val_preds, labels=val_unique_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_target_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.title('1D CNN - Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    cm_path = os.path.join("processed_data", "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to '{cm_path}'")

    # 7. Model Comparison Against Random Forest Baseline
    rf_acc, rf_p_macro, rf_r_macro, rf_f1_macro, rf_f1_weighted = 26.19, 9.69, 13.61, 10.24, 24.86

    print("\n" + "=" * 75)
    print("       MODEL COMPARISON: 1D CNN vs. RANDOM FOREST BASELINE       ")
    print("=" * 75)
    print(f"{'Metric':<22} | {'Random Forest':<15} | {'1D CNN':<15} | {'Improvement':<15}")
    print("-" * 75)
    print(f"{'Validation Accuracy':<22} | {rf_acc:>14.2f}% | {acc * 100:>14.2f}% | {(acc * 100 - rf_acc):>+14.2f}%")
    print(f"{'Macro Precision':<22} | {rf_p_macro:>14.2f}% | {p_macro * 100:>14.2f}% | {(p_macro * 100 - rf_p_macro):>+14.2f}%")
    print(f"{'Macro Recall':<22} | {rf_r_macro:>14.2f}% | {r_macro * 100:>14.2f}% | {(r_macro * 100 - rf_r_macro):>+14.2f}%")
    print(f"{'Macro F1-Score':<22} | {rf_f1_macro:>14.2f}% | {f1_macro * 100:>14.2f}% | {(f1_macro * 100 - rf_f1_macro):>+14.2f}%")
    print(f"{'Weighted F1-Score':<22} | {rf_f1_weighted:>14.2f}% | {f1_weighted * 100:>14.2f}% | {(f1_weighted * 100 - rf_f1_weighted):>+14.2f}%")
    print("-" * 75)

    if acc * 100 > rf_acc:
        print(" VERDICT: The 1D CNN IMPROVES over the Random Forest baseline!")
    else:
        print(" VERDICT: The 1D CNN does NOT outperform the Random Forest baseline.")

    print("\n" + "=" * 75)
    print("⚠️ MEDICAL DISCLAIMER: This deep learning model is strictly built for")
    print("research, education, and hackathon benchmarking purposes.")
    print("It is NOT intended, validated, or certified for clinical diagnosis.")
    print("=" * 75)

if __name__ == "__main__":
    train_and_evaluate_cnn()
