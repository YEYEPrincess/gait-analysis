import numpy as np
import pandas as pd
import os
from pathlib import Path

def is_default_frame(arr, threshold=50000):
    return np.all(arr == threshold)

CSV_PATH = Path(__file__).resolve().parent / "common" / "insole_frames_flat.csv"
df = pd.read_csv(CSV_PATH)

# Separate reference (client 0) and crosstalk (client 1)
ref_df = df[df["client"] == 0].reset_index(drop=True)   # reference
ct_df  = df[df["client"] == 1].reset_index(drop=True)   # crosstalk

# Extract timestamps
ref_t = ref_df["timestamp"].to_numpy()
ct_t  = ct_df["timestamp"].to_numpy()

# Extract flattened frames ch_0...ch_252
ref_vals = ref_df.filter(like="ch_").to_numpy(dtype=np.float32)
ct_vals  = ct_df.filter(like="ch_").to_numpy(dtype=np.float32)

# Stop if there is no frame
if len(ref_t) == 0 or len(ct_t) == 0:
    raise SystemExit("Missing ref or ct frames")

# Lists that will contain matched pairs
paired_ref = []
paired_ct  = []

# For each ref_t, find insertion point in ct_t
# idx is the index of the first ct_t >= ref_t
idx = np.searchsorted(ct_t, ref_t, side="left")  # shape (Nref,)

# Candidates: ct before (idx-1) and ct after (idx)
idx_before = idx - 1
idx_after  = idx

# Validity of candidates
valid_before = (idx_before >= 0)
valid_after  = (idx_after < len(ct_t))

# Compute dt for both candidates, set invalid to +inf
dt_before = np.full_like(ref_t, np.inf, dtype=np.float64)
dt_after  = np.full_like(ref_t, np.inf, dtype=np.float64)

dt_before[valid_before] = np.abs(ct_t[idx_before[valid_before]] - ref_t[valid_before])
dt_after[valid_after]   = np.abs(ct_t[idx_after[valid_after]]   - ref_t[valid_after])

# Pick the closer candidate
pick_after = dt_after < dt_before
best_idx = np.where(pick_after, idx_after, idx_before)
best_dt  = np.minimum(dt_before, dt_after)

# Keep only pairs within MAX_DELTA
MAX_DELTA=0.02
keep = best_dt <= MAX_DELTA

paired_ref = ref_vals[keep]
paired_ct  = ct_vals[best_idx[keep]]

print("Final paired shapes:", paired_ref.shape, paired_ct.shape)

# Create mask to remove frame pairs where no sensor is active
mask_ref  = ~np.apply_along_axis(is_default_frame, 1, paired_ref)
mask_ct = ~np.apply_along_axis(is_default_frame, 1, paired_ct)

good_mask = mask_ref & mask_ct

# Stop the programme if no frame pair is useful
if not good_mask.any():
    print("No training data")
    raise SystemExit

# Keep useful frames for training
filtered_ref  = paired_ref[good_mask]
filtered_ct = paired_ct[good_mask]

# ---------- Rebuild 2D matrices (33x15) from filtered flat vectors ----------

image_pattern = np.array([
    [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
], dtype=int)

# Sensor index positions (253 active locations)
indices = np.argwhere(image_pattern == 1)
ii, jj = indices[:, 0], indices[:, 1]

N = filtered_ref.shape[0]
rows, cols = 33, 15

# Allocate 2D matrices
ref_mats = np.zeros((N, rows, cols), dtype=np.float32)
ct_mats  = np.zeros((N, rows, cols), dtype=np.float32)

# Fill only active sensor locations
ref_mats[:, ii, jj] = filtered_ref
ct_mats[:,  ii, jj] = filtered_ct

print("2D shapes:", ref_mats.shape, ct_mats.shape)

# ---------- Normalize to [0, 1] ----------
MAX_R = 50000.0
ref_norm = np.clip(ref_mats, 0.0, MAX_R) / MAX_R
ct_norm  = np.clip(ct_mats,  0.0, MAX_R) / MAX_R

# ---------- Mark non-sensor/faulty zones as -1 ----------
active_mask = (image_pattern == 1)       # (33, 15) True = sensor, False = empty
inactive_mask = ~active_mask

inactive_mask[0][7] = True
inactive_mask[1][10] = True
inactive_mask[3][4] = True
inactive_mask[4][11] = True
#print(inactive_mask.astype(int))

# Broadcast mask over batch dimension
ref_norm[:, inactive_mask] = -1.0
ct_norm[:,  inactive_mask] = -1.0

# invert sensor values: 1 = pressed, 0 = not pressed
ref_norm = np.where(ref_norm >= 0, 1.0 - ref_norm, -1.0)
ct_norm  = np.where(ct_norm  >= 0, 1.0 - ct_norm, -1.0)

# ---------- Final ML format: (N, 1, 33, 15) ----------
X = ct_norm[:,  None, :, :]   # crosstalk
Y = ref_norm[:, None, :, :]   # reference

print("Final X, Y shapes:", X.shape, Y.shape)   

# ---------- Save NPZ ----------
npz_path = "data_ct"
OUTPUT_FILENAME = "sample_ct_dataset.npz"
os.makedirs(npz_path, exist_ok=True)
np.savez(os.path.join(npz_path, OUTPUT_FILENAME), X=X, Y=Y)
