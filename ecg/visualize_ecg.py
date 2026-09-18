import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_ecg_signals(num_classes_to_plot=4, points_to_show=1000):
    dataset_path = os.path.join("processed_data", "ecg_dataset.npz")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Processed dataset not found at '{dataset_path}'. Please run preprocess.py first.")
        return

    # Load preprocessed dataset
    data = np.load(dataset_path)
    X_train = data['X_train']
    y_train = data['y_train']
    class_names = data['class_names']

    print("=" * 60)
    print("               ECG SIGNAL VISUALIZATION                ")
    print("=" * 60)
    print(f"Loaded training dataset: {X_train.shape[0]} samples, {X_train.shape[1]} length")

    # Select 4 representative classes
    selected_class_indices = [0, 3, 6, 13]  # 1 NSR, 4 AFIB, 7 PVC, 14 LBBBB
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(len(selected_class_indices), 1, figsize=(14, 10), sharex=True)
    fig.suptitle('ECG Arrhythmia Signal Samples Across Different Rhythm Classes', fontsize=16, fontweight='bold', y=0.98)

    time_axis = np.arange(points_to_show)

    for i, class_idx in enumerate(selected_class_indices):
        class_name = class_names[class_idx]
        
        # Find first sample matching this class
        match_idx = np.where(y_train == class_idx)[0][0]
        signal = X_train[match_idx, :points_to_show]

        ax = axes[i]
        ax.plot(time_axis, signal, color=colors[i % len(colors)], linewidth=1.2, label=f"Class {class_idx}: {class_name}")
        ax.set_title(f"Class {class_idx}: {class_name}", fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel("Normalized Amplitude (Z-Score)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Sample Index (First 1,000 points of 3,600)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_fig_path = os.path.join("processed_data", "ecg_visualization.png")
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved successfully to '{output_fig_path}'")
    print("=" * 60)

if __name__ == "__main__":
    visualize_ecg_signals()
