import numpy as np
import glob
import re
import os

# Path to folder containing datasetX.npz files
joint_estimation = True
if joint_estimation:
    DATA_DIR = "data_for_3djoint"
else:
    DATA_DIR = "data_ct"

# Pattern to match dataset1.npz, dataset2.npz, ..., dataset99.npz
pattern = os.path.join(DATA_DIR, "dataset*.npz")
npz_files = glob.glob(pattern)

# Keep only files with a numeric suffix > 0
npz_files = [f for f in npz_files if re.match(r".*dataset[0-9]\d*\.npz$", f)]

if len(npz_files) == 0:
    raise RuntimeError(f"No datasetX.npz files found in {DATA_DIR}/ (X > 0).")

print("Found NPZ files:")
for f in npz_files:
    print("  -", f)

# Containers
X_list = []
Y_list = []

# Sort files numerically instead of lexicographically
def extract_number(filename):
    return int(re.findall(r"\d+", os.path.basename(filename))[0])

# Keep only datasetX.npz with X > 6
npz_files = [f for f in npz_files if extract_number(f) > 6]

npz_files_sorted = sorted(npz_files, key=extract_number)

# Load and accumulate arrays
for f in npz_files_sorted:
    print(f"\nLoading {f} ...")
    data = np.load(f)

    if "X" not in data or "Y" not in data:
        raise ValueError(f"{f} does not contain X and Y arrays!")

    X = data["X"]
    Y = data["Y"]

    print(f"  Shapes → X: {X.shape}, Y: {Y.shape}")

    X_list.append(X)
    Y_list.append(Y)

# Concatenate along batch dimension N
X_merged = np.concatenate(X_list, axis=0)
Y_merged = np.concatenate(Y_list, axis=0)

print("\nFinal merged shapes:")
print("  X_merged:", X_merged.shape)
print("  Y_merged:", Y_merged.shape)

# Save merged dataset
output_path = os.path.join(DATA_DIR, "dataset_merged_b.npz")
np.savez(output_path, X=X_merged, Y=Y_merged)

print(f"\nMerged dataset saved as: {output_path}")
