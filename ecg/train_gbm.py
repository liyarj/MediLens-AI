import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def train_and_evaluate_gbm():
    input_path = os.path.join("processed_data", "features_dataset.npz")
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Feature dataset not found at '{input_path}'. Run feature_extraction.py first.")

    print("=" * 75)
    print("      GRADIENT BOOSTED TREES MODEL: HistGradientBoostingClassifier      ")
    print("=" * 75)

    # 1. Load extracted features and labels
    data = np.load(input_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    class_names = [str(c) for c in data['class_names']]
    feature_names = [str(f) for f in data['feature_names']]

    total_features = len(feature_names)
    num_classes = len(class_names)

    print(f"\n[1] Extracted Features Specification:")
    print(f" - Total Features            : {total_features}")
    print(f" - Training Matrix Shape     : {X_train.shape}")
    print(f" - Validation Matrix Shape   : {X_val.shape}")

    # 2. Check class distribution & absent classes
    train_classes_present = set(np.unique(y_train))
    val_classes_present = set(np.unique(y_val))
    absent_classes_val = train_classes_present.difference(val_classes_present)

    print("\n--- [A] CLASS DISTRIBUTION & GROUPING REPORT ---")
    print(f"Training Active Classes     : {len(train_classes_present)}")
    print(f"Validation Active Classes   : {len(val_classes_present)}")

    if absent_classes_val:
        print("Note: Due to recording-level grouping (0 data leakage), the following classes exist ONLY in Train:")
        for cid in sorted(list(absent_classes_val)):
            print(f" - Class {cid}: '{class_names[cid]}'")

    # 3. Train HistGradientBoostingClassifier
    print("\n[2] Training HistGradientBoostingClassifier (class_weight='balanced', random_state=42)...")
    gbm = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=31,
        class_weight='balanced',
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    # Fit ONLY on training data
    gbm.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(model_dir, "hist_gradient_boosting.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(gbm, f)
    print(f"Saved trained HistGradientBoosting model to '{model_path}'")

    # 4. Evaluate on Validation Set
    print("\n[3] Evaluating Model on Validation Set (Zero-Leakage Group Split)...")
    y_val_pred = gbm.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_val, y_val_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 75)
    print("               HIST-GRADIENT BOOSTING VALIDATION METRICS               ")
    print("=" * 75)
    print(f" Validation Accuracy        : {acc * 100:.2f}%")
    print(f" Macro Precision            : {p_macro * 100:.2f}%")
    print(f" Macro Recall               : {r_macro * 100:.2f}%")
    print(f" Macro F1-Score             : {f1_macro * 100:.2f}%")
    print(f" Weighted F1-Score          : {f1_weighted * 100:.2f}%")
    print("=" * 75)

    # Filter evaluation for unique classes present in validation set
    val_unique_labels = sorted(list(val_classes_present))
    val_target_names = [class_names[cid] for cid in val_unique_labels]

    print("\n--- [B] PER-CLASS CLASSIFICATION REPORT (Validation Set) ---")
    report = classification_report(
        y_val, y_val_pred,
        labels=val_unique_labels,
        target_names=val_target_names,
        zero_division=0
    )
    print(report)

    # 5. Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_val, y_val_pred, labels=val_unique_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_target_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.title('HistGradientBoosting - Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()

    cm_path = os.path.join("processed_data", "gbm_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to '{cm_path}'")

    # 6. Feature Importance via Permutation Importance on Training Set
    print("\n[4] Calculating Permutation Feature Importance...")
    perm_importance = permutation_importance(gbm, X_train, y_train, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_importances_idx = perm_importance.importances_mean.argsort()[::-1][:10]

    print("\n--- [C] TOP 10 MOST IMPORTANT FEATURES (Permutation Importance) ---")
    for rank, idx in enumerate(sorted_importances_idx, 1):
        score = perm_importance.importances_mean[idx]
        print(f" {rank:2d}. {feature_names[idx]:<25} : {score:.4f}")

    # 7. Comparative Analysis Table across all 3 models
    rf_acc, rf_p_macro, rf_r_macro, rf_f1_macro, rf_f1_weighted = 26.19, 9.69, 13.61, 10.24, 24.86
    cnn_acc, cnn_p_macro, cnn_r_macro, cnn_f1_macro, cnn_f1_weighted = 9.05, 7.07, 9.66, 2.86, 4.91

    print("\n" + "=" * 75)
    print("      TRI-MODEL COMPARISON: RANDOM FOREST vs 1D CNN vs HIST-GBM      ")
    print("=" * 75)
    print(f"{'Metric':<22} | {'Random Forest':<14} | {'1D CNN':<10} | {'Hist-GBM':<12} | {'Best Model':<12}")
    print("-" * 75)
    print(f"{'Validation Accuracy':<22} | {rf_acc:>13.2f}% | {cnn_acc:>9.2f}% | {acc * 100:>11.2f}% | {'Hist-GBM' if acc*100 > rf_acc else ('RF' if rf_acc > cnn_acc else 'CNN')}")
    print(f"{'Macro Precision':<22} | {rf_p_macro:>13.2f}% | {cnn_p_macro:>9.2f}% | {p_macro * 100:>11.2f}% | {'Hist-GBM' if p_macro*100 > rf_p_macro else 'RF'}")
    print(f"{'Macro Recall':<22} | {rf_r_macro:>13.2f}% | {cnn_r_macro:>9.2f}% | {r_macro * 100:>11.2f}% | {'Hist-GBM' if r_macro*100 > rf_r_macro else 'RF'}")
    print(f"{'Macro F1-Score':<22} | {rf_f1_macro:>13.2f}% | {cnn_f1_macro:>9.2f}% | {f1_macro * 100:>11.2f}% | {'Hist-GBM' if f1_macro*100 > rf_f1_macro else 'RF'}")
    print(f"{'Weighted F1-Score':<22} | {rf_f1_weighted:>13.2f}% | {cnn_f1_weighted:>9.2f}% | {f1_weighted * 100:>11.2f}% | {'Hist-GBM' if f1_weighted*100 > rf_f1_weighted else 'RF'}")
    print("-" * 75)

    if acc * 100 > rf_acc:
        print(" VERDICT: HistGradientBoosting IMPROVES over the Random Forest baseline!")
    else:
        print(f" VERDICT: Random Forest remains the top baseline ({rf_acc:.2f}% vs {acc*100:.2f}%).")

    print("\n" + "=" * 75)
    print("⚠️ MEDICAL DISCLAIMER: This machine learning model is strictly built")
    print("for research, education, and hackathon benchmarking purposes.")
    print("It is NOT intended or certified for clinical diagnostic use.")
    print("=" * 75)

if __name__ == "__main__":
    train_and_evaluate_gbm()
