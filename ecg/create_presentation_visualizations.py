import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_visualizations():
    processed_dir = "processed_data"
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load Extracted Features Dataset & Model
    feats_data = np.load(os.path.join(processed_dir, "features_dataset.npz"))
    X_train_feats = feats_data['X_train']
    X_val_feats = feats_data['X_val']
    y_train = feats_data['y_train']
    y_val = feats_data['y_val']
    class_names = [str(c) for c in feats_data['class_names']]
    feature_names = [str(f) for f in feats_data['feature_names']]

    model_path = os.path.join("model", "hist_gradient_boosting.pkl")
    with open(model_path, "rb") as f:
        gbm = pickle.load(f)

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    # =========================================================================
    # A. Feature Correlation Heatmap (43 Features)
    # =========================================================================
    print("Generating Feature-Correlation Heatmap...")
    df_feats = pd.DataFrame(X_train_feats, columns=feature_names)
    corr_matrix = df_feats.corr()

    plt.figure(figsize=(16, 14))
    heatmap = sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        vmin=-1.0, vmax=1.0,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
    )
    plt.title("ECG Feature Correlation Matrix (43 Extracted Features)", fontsize=16, fontweight='bold', pad=15)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    corr_path = os.path.join(processed_dir, "feature_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {corr_path}")

    # =========================================================================
    # B. Feature Importance Visualization (Top 15 Permutation Importances)
    # =========================================================================
    print("Generating Feature Importance Bar Chart...")
    perm_importance = permutation_importance(gbm, X_train_feats, y_train, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1][:15]

    top_feats = [feature_names[i] for i in sorted_idx][::-1]
    top_scores = [perm_importance.importances_mean[i] for i in sorted_idx][::-1]

    plt.figure(figsize=(10, 7))
    bars = plt.barh(top_feats, top_scores, color=sns.color_palette("mako", len(top_feats)))
    plt.title("Top 15 Feature Importances (HistGradientBoosting Permutation Scores)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Mean Permutation Importance Score", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Annotate bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                 va='center', ha='left', fontsize=9, fontweight='bold')

    plt.tight_layout()
    imp_path = os.path.join(processed_dir, "feature_importance.png")
    plt.savefig(imp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {imp_path}")

    # =========================================================================
    # C. Model Comparison Chart (Random Forest vs 1D CNN vs Hist-GBM)
    # =========================================================================
    print("Generating Model Comparison Chart...")
    metrics_data = {
        'Metric': ['Validation Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score', 'Weighted F1-Score'],
        'Random Forest Baseline': [26.19, 9.69, 13.61, 10.24, 24.86],
        '1D CNN Model': [9.05, 7.07, 9.66, 2.86, 4.91],
        'HistGradientBoosting': [38.10, 23.93, 19.12, 19.00, 37.12]
    }
    df_comp = pd.DataFrame(metrics_data)
    df_melted = df_comp.melt(id_vars='Metric', var_name='Model', value_name='Percentage (%)')

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_melted,
        x='Metric',
        y='Percentage (%)',
        hue='Model',
        palette=['#4c72b0', '#c44e52', '#55a868']
    )
    plt.title("Tri-Model Benchmark Comparison on Zero-Leakage Group Split", fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("Performance Score (%)", fontsize=12)
    plt.xlabel("")
    plt.ylim(0, 50)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Machine Learning Model", frameon=True, loc='upper left')

    # Annotate values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height + 0.8),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    comp_path = os.path.join(processed_dir, "model_comparison_chart.png")
    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {comp_path}")

    # =========================================================================
    # D. Presentation-Ready Confusion Matrix
    # =========================================================================
    print("Generating Presentation-Ready Final Confusion Matrix...")
    y_val_pred = gbm.predict(X_val_feats)
    val_unique_labels = sorted(list(set(y_val)))
    val_target_names = [class_names[cid] for cid in val_unique_labels]

    cm = confusion_matrix(y_val, y_val_pred, labels=val_unique_labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_target_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.title('Final Champion Model (HistGradientBoosting) — Validation Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    final_cm_path = os.path.join(processed_dir, "final_confusion_matrix.png")
    plt.savefig(final_cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {final_cm_path}")

    print("All presentation-ready visualizations generated successfully!")

if __name__ == "__main__":
    generate_visualizations()
