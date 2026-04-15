import os

import numpy as np
import matplotlib.pyplot as plt
import math
# ---------- Load and preprocess data ----------
# Load raw data
DATA_DIR = "data_ct"
DATASET_FILENAME = "sample_ct_dataset.npz"

npz_path = os.path.join(DATA_DIR, DATASET_FILENAME)
data = np.load(npz_path)
X = data["X"]   # Crosstalk  -> (N, 1, 33, 15)
Y = data["Y"]   # Reference  -> (N, 1, 33, 15)

# Number of frames to display
n = min(40,X.shape[0])
npercol = 10
ncol= (n//npercol + 1)*2

fig, axes = plt.subplots(npercol, ncol, figsize=(6, 3*n))

for i in range(n):
    ct = X[i+100, 0]   # (33, 15)
    ref = Y[i+100, 0]  # (33, 15)

    col = (i//npercol) * 2

    # Crosstalk (X)
    axes[i % npercol, col].imshow(ct, cmap='viridis', vmin=0, vmax=1)
    axes[i% npercol, col].set_title(f"CT #{i}")
    axes[i% npercol, col].axis("off")

    # Reference (Y)
    axes[i% npercol, col + 1].imshow(ref, cmap='viridis', vmin=0, vmax=1)
    axes[i% npercol, col + 1].set_title(f"REF #{i}")
    axes[i% npercol, col + 1].axis("off")

fig.subplots_adjust(
    left=0.01, right=0.99,
    top=0.99, bottom=0.01,
    wspace=0.02, hspace=0.02
)

plt.show()
