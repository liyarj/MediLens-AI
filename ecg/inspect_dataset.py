import os
import glob
from collections import defaultdict
import scipy.io

def inspect_dataset():
    dataset_dir = "MLII"
    print("=" * 60)
    print("           ECG ARRHYTHMIA DATASET INSPECTION           ")
    print("=" * 60)

    # 1 & 2: Recursively find all .mat files and identify class names
    mat_pattern = os.path.join(dataset_dir, "**", "*.mat")
    mat_files = sorted(glob.glob(mat_pattern, recursive=True))

    print(f"\n[1] Finding .mat files inside '{dataset_dir}/'...")
    print(f"Total .mat files found: {len(mat_files)}")

    class_files = defaultdict(list)
    for filepath in mat_files:
        rel_path = os.path.relpath(filepath, dataset_dir)
        parts = rel_path.split(os.sep)
        if len(parts) > 1:
            class_name = parts[0]
        else:
            class_name = "Root"
        class_files[class_name].append(filepath)

    # 7: Rhythm classes count
    num_classes = len(class_files)
    print(f"\n[7] Total ECG Rhythm Classes Found: {num_classes}")

    # 3: File counts per class
    print("\n[3] File counts per class:")
    print("-" * 45)
    print(f"{'Class / Folder Name':<25} | {'Count':<10}")
    print("-" * 45)
    for class_name, files in sorted(class_files.items()):
        print(f"{class_name:<25} | {len(files):<10}")
    print("-" * 45)

    # 6: 3 Example filenames from each class
    print("\n[6] Example filenames per class (3 per class):")
    print("-" * 60)
    for class_name, files in sorted(class_files.items()):
        examples = [os.path.basename(f) for f in files[:3]]
        print(f"Class '{class_name}': {', '.join(examples)}")
    print("-" * 60)

    # 4 & 5: Inspect .mat variable names, signal shape, data type, number of samples, & length consistency
    print("\n[4 & 5] Inspecting .mat structure and signal length consistency...")

    shapes_encountered = set()
    dtypes_encountered = set()
    signal_keys_encountered = set()
    lengths = set()

    sampled_details = []
    # Sample up to 5 files per class for detailed inspection
    all_inspected_files = []
    for class_name, files in sorted(class_files.items()):
        all_inspected_files.extend(files)

    signal_length_consistent = True

    for i, filepath in enumerate(all_inspected_files):
        try:
            mat_data = scipy.io.loadmat(filepath)
            # Filter out metadata keys starting with '__'
            var_keys = [k for k in mat_data.keys() if not k.startswith("__")]

            for key in var_keys:
                signal_keys_encountered.add(key)
                val = mat_data[key]
                shapes_encountered.add(val.shape)
                dtypes_encountered.add(str(val.dtype))
                length = val.size
                lengths.add(length)

                if i < 5:  # Store first 5 files as examples
                    sampled_details.append({
                        "filename": os.path.basename(filepath),
                        "class": os.path.dirname(os.path.relpath(filepath, dataset_dir)),
                        "var_name": key,
                        "shape": val.shape,
                        "dtype": str(val.dtype),
                        "num_samples": length
                    })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    print("\nSample .mat File Detailed Breakdown (First 5 inspected files):")
    print("-" * 80)
    print(f"{'Filename':<20} | {'Var Name':<10} | {'Shape':<12} | {'Data Type':<10} | {'Num Samples':<12}")
    print("-" * 80)
    for detail in sampled_details:
        print(f"{detail['filename']:<20} | {detail['var_name']:<10} | {str(detail['shape']):<12} | {detail['dtype']:<10} | {detail['num_samples']:<12}")
    print("-" * 80)

    # Check signal length consistency across all files
    if len(lengths) == 1:
        signal_length_consistent = True
        unique_len = next(iter(lengths))
        length_msg = f"YES - All signals have the EXACT same length ({unique_len} samples)."
    else:
        signal_length_consistent = False
        length_msg = f"NO - Multiple signal lengths found: {sorted(list(lengths))}"

    print(f"\nLength Consistency Check: {length_msg}")

    # 8: Dataset Summary
    print("\n" + "=" * 60)
    print("                    DATASET SUMMARY                    ")
    print("=" * 60)
    print(f" Total .mat files      : {len(mat_files)}")
    print(f" Total Rhythm Classes  : {num_classes}")
    print(f" MATLAB Variable Name  : {', '.join(sorted(signal_keys_encountered)) if signal_keys_encountered else 'N/A'}")
    print(f" Signal Shape(s)       : {list(shapes_encountered)}")
    print(f" Data Type(s)          : {list(dtypes_encountered)}")
    print(f" Constant Length?      : {'Yes' if signal_length_consistent else 'No'}")
    if signal_length_consistent:
        print(f" Signal Length         : {next(iter(lengths))} samples per ECG record")
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()
