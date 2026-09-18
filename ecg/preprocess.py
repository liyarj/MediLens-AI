import os
import glob
import json
import numpy as np
import scipy.io
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold

def extract_recording_id(filepath):
    """
    Extracts recording ID from filename e.g.:
    '100m (0).mat' -> '100m'
    '100m (1).mat' -> '100m'
    """
    basename = os.path.basename(filepath)
    if " (" in basename:
        return basename.split(" (")[0]
    return basename.split(".")[0]

def load_and_preprocess_dataset():
    dataset_dir = "MLII"
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("        ECG PREPROCESSING & GROUPED STRATIFIED SPLIT REPORT        ")
    print("=" * 70)

    # 1. Discover classes from folder structure
    print(f"\n[1] Discovering class directories in '{dataset_dir}/'...")
    subdirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ], key=lambda x: int(x.split()[0]) if x.split()[0].isdigit() else x)

    # Label mapping (class_name -> integer ID)
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(subdirs)}
    id_to_class = {idx: cls_name for idx, cls_name in enumerate(subdirs)}

    file_paths = []
    labels = []
    recording_ids = []
    class_rec_map = defaultdict(set)

    for cls_name in subdirs:
        cls_dir = os.path.join(dataset_dir, cls_name)
        mat_files = sorted(glob.glob(os.path.join(cls_dir, "*.mat")))
        for fpath in mat_files:
            file_paths.append(fpath)
            labels.append(class_to_id[cls_name])
            rec_id = extract_recording_id(fpath)
            recording_ids.append(rec_id)
            class_rec_map[cls_name].add(rec_id)

    total_samples = len(file_paths)
    unique_recordings = sorted(list(set(recording_ids)))
    total_unique_recs = len(unique_recordings)

    print("\n--- [A] LABEL MAPPING & RECORDING DISTRIBUTION ---")
    print("Label Mapping:")
    print(json.dumps(class_to_id, indent=4))
    print(f"\nTotal Samples           : {total_samples}")
    print(f"Total Unique Recordings : {total_unique_recs}")

    # 2. Extract signals and normalize
    print("\n[2] Extracting 'val' arrays and performing Z-score normalization...")
    signal_length = 3600
    X = np.zeros((total_samples, signal_length), dtype=np.float32)

    for i, fpath in enumerate(file_paths):
        mat_data = scipy.io.loadmat(fpath)
        raw_signal = mat_data['val'].squeeze().astype(np.float32)

        if raw_signal.shape[0] != signal_length:
            raw_signal = raw_signal.flatten()[:signal_length]

        # Z-score normalization: (signal - mean) / (std + 1e-8)
        mean_val = np.mean(raw_signal)
        std_val = np.std(raw_signal)
        X[i] = (raw_signal - mean_val) / (std_val + 1e-8)

    y = np.array(labels, dtype=np.int64)
    rec_ids_arr = np.array(recording_ids)

    # 3. Grouped Train/Validation Split using StratifiedGroupKFold
    print("\n[3] Performing GROUPED Stratified Train/Validation Split (80/20 target)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(X, y, groups=rec_ids_arr))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    files_train = [file_paths[i] for i in train_idx]
    files_val = [file_paths[i] for i in val_idx]
    recs_train = rec_ids_arr[train_idx]
    recs_val = rec_ids_arr[val_idx]

    train_unique_recs = set(recs_train)
    val_unique_recs = set(recs_val)
    overlap_recs = train_unique_recs.intersection(val_unique_recs)

    # 4. Reporting results
    print("\n" + "=" * 70)
    print("                    DATASET SPLIT SUMMARY                    ")
    print("=" * 70)
    print(f" Total Processed Samples               : {total_samples}")
    print(f" Training Samples                      : {len(X_train)} ({len(X_train)/total_samples*100:.1f}%)")
    print(f" Validation Samples                    : {len(X_val)} ({len(X_val)/total_samples*100:.1f}%)")
    print(f" Unique Recordings in Training        : {len(train_unique_recs)}")
    print(f" Unique Recordings in Validation      : {len(val_unique_recs)}")
    print(f" Recording-ID Overlap Count            : {len(overlap_recs)}")

    print("\n--- [CONFIRMATION OF ZERO OVERLAP] ---")
    if len(overlap_recs) == 0:
        print(" SUCCESS: ZERO recording-ID overlap between train and validation splits!")
        print(" Zero Data Leakage: All fragments from each recording ID belong strictly to Train or Val.")
    else:
        print(f" WARNING: Overlap detected! Shared IDs: {overlap_recs}")

    print("\n--- [CLASS DISTRIBUTION IN TRAIN & VALIDATION] ---")
    train_counts = np.bincount(y_train, minlength=len(subdirs))
    val_counts = np.bincount(y_val, minlength=len(subdirs))

    print("-" * 70)
    print(f"{'Class Name':<20} | {'Total':<6} | {'Train Count':<11} | {'Val Count':<10} | {'Status':<15}")
    print("-" * 70)

    rare_classes_warn = []
    for cls_name in subdirs:
        cid = class_to_id[cls_name]
        tot = train_counts[cid] + val_counts[cid]
        t_cnt = train_counts[cid]
        v_cnt = val_counts[cid]
        
        if t_cnt > 0 and v_cnt > 0:
            status = "BOTH"
        elif t_cnt > 0 and v_cnt == 0:
            status = "TRAIN ONLY"
            rare_classes_warn.append((cls_name, len(class_rec_map[cls_name])))
        elif t_cnt == 0 and v_cnt > 0:
            status = "VAL ONLY"
            rare_classes_warn.append((cls_name, len(class_rec_map[cls_name])))

        print(f"{cls_name:<20} | {tot:<6} | {t_cnt:<11} | {v_cnt:<10} | {status:<15}")
    print("-" * 70)

    if rare_classes_warn:
        print("\n--- [HONEST REPORTING ON RARE CLASSES] ---")
        print("Note: The following classes appear in only one split due to group isolation:")
        for cname, num_recs in rare_classes_warn:
            print(f" - '{cname}': Only {num_recs} unique recording ID(s) exist across dataset. Kept intact in one split to prevent data leakage.")

    # 5. Save corrected dataset
    output_npz = os.path.join(output_dir, "ecg_dataset.npz")
    np.savez_compressed(
        output_npz,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        class_names=np.array(subdirs),
        files_train=np.array(files_train),
        files_val=np.array(files_val),
        recs_train=recs_train,
        recs_val=recs_val
    )

    label_map_path = os.path.join(output_dir, "label_mapping.json")
    with open(label_map_path, "w") as f:
        json.dump({
            "class_to_id": class_to_id,
            "id_to_class": {str(k): v for k, v in id_to_class.items()}
        }, f, indent=4)

    print(f"\nSuccessfully saved corrected processed dataset to '{output_npz}'")
    print(f"Saved updated label mapping to '{label_map_path}'")
    print("=" * 70)

if __name__ == "__main__":
    load_and_preprocess_dataset()
