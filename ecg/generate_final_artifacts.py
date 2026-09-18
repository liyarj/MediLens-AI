import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def generate_final_artifacts():
    dataset_path = os.path.join("processed_data", "features_dataset.npz")
    model_path = os.path.join("model", "hist_gradient_boosting.pkl")

    data = np.load(dataset_path)
    X_val = data['X_val']
    y_val = data['y_val']
    class_names = [str(c) for c in data['class_names']]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_val_pred = model.predict(X_val)

    val_unique_labels = sorted(list(set(y_val)))
    val_target_names = [class_names[cid] for cid in val_unique_labels]

    # Save final classification report text file
    report_str = classification_report(
        y_val, y_val_pred,
        labels=val_unique_labels,
        target_names=val_target_names,
        zero_division=0
    )

    report_path = os.path.join("processed_data", "final_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=========================================================\n")
        f.write("      FINAL EVALUATION CLASSIFICATION REPORT (VAL SET)    \n")
        f.write("=========================================================\n\n")
        f.write(report_str)

    # Save final confusion matrix image
    cm = confusion_matrix(y_val, y_val_pred, labels=val_unique_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_target_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.title('Final Model (HistGradientBoosting) - Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()

    cm_path = os.path.join("processed_data", "final_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("Successfully generated final confusion matrix and classification report artifacts.")

if __name__ == "__main__":
    generate_final_artifacts()
