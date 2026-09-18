import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def train_and_evaluate_baseline():
    input_path = os.path.join("processed_data", "features_dataset.npz")
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Feature dataset not found at '{input_path}'. Run feature_extraction.py first.")

    print("=" * 70)
    print("         TRADITIONAL ML BASELINE: RANDOM FOREST CLASSIFIER         ")
    print("=" * 70)

    # 1. Load extracted features and labels
    data = np.load(input_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    class_names = [str(c) for c in data['class_names']]
    feature_names = [str(f) for f in data['feature_names']]

    print(f"\n[1] Loaded Extracted Features:")
    print(f" - Train Feature Matrix Shape: {X_train.shape}")
    print(f" - Val Feature Matrix Shape  : {X_val.shape}")
    print(f" - Number of Features        : {len(feature_names)}")

    # 2. Check class presence in Train vs Validation
    train_classes_present = set(np.unique(y_train))
    val_classes_present = set(np.unique(y_val))
    absent_classes_val = train_classes_present.difference(val_classes_present)

    print("\n--- [A] ABSENT CLASSES IN VALIDATION REPORT ---")
    if absent_classes_val:
        print("Note: Due to recording-level grouping (0 data leakage), the following classes exist ONLY in Train:")
        for cid in sorted(list(absent_classes_val)):
            print(f" - Class {cid}: '{class_names[cid]}'")
    else:
        print(" All classes are present in both Train and Validation sets.")

    # 3. Train Random Forest Classifier
    print("\n[2] Training Random Forest Classifier (n_estimators=100, class_weight='balanced', random_state=42)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    # Fit ONLY on training data
    clf.fit(X_train, y_train)

    # 4. Save trained model to model/random_forest.pkl
    model_path = os.path.join(model_dir, "random_forest.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved trained Random Forest model to '{model_path}'")

    # 5. Evaluate on Validation Set
    print("\n[3] Evaluating Model on Validation Set (Zero-Leakage Test)...")
    y_val_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_val, y_val_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 70)
    print("                  BASELINE MODEL EVALUATION METRICS                  ")
    print("=" * 70)
    print(f" Validation Accuracy        : {acc * 100:.2f}%")
    print(f" Precision (Macro / Weighted): {p_macro * 100:.2f}% / {p_weighted * 100:.2f}%")
    print(f" Recall    (Macro / Weighted): {r_macro * 100:.2f}% / {r_weighted * 100:.2f}%")
    print(f" F1-Score  (Macro / Weighted): {f1_macro * 100:.2f}% / {f1_weighted * 100:.2f}%")
    print("=" * 70)

    # Filter evaluation for unique classes present in validation set
    val_unique_labels = sorted(list(val_classes_present))
    val_class_target_names = [class_names[cid] for cid in val_unique_labels]

    print("\n--- [B] PER-CLASS CLASSIFICATION REPORT (Validation Set) ---")
    report = classification_report(
        y_val, y_val_pred,
        labels=val_unique_labels,
        target_names=val_class_target_names,
        zero_division=0
    )
    print(report)

    # 6. Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_val, y_val_pred, labels=val_unique_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_class_target_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.title('Baseline Random Forest - Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()

    cm_path = os.path.join("processed_data", "baseline_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to '{cm_path}'")

    # Feature Importance Top 10
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    print("\n--- [C] TOP 10 MOST IMPORTANT FEATURES ---")
    for rank, idx in enumerate(top_indices, 1):
        print(f" {rank:2d}. {feature_names[idx]:<22} : {importances[idx]:.4f}")

    print("\n" + "=" * 70)
    print("⚠️ MEDICAL DISCLAIMER: This machine learning model is strictly built")
    print("for research, education, and hackathon benchmarking purposes.")
    print("It is NOT intended or certified for clinical or medical diagnostic use.")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate_baseline()
