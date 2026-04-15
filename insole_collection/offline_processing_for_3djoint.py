import numpy as np
import pandas as pd
import os
from pathlib import Path

def is_default_frame(arr, threshold=50000):
    return np.all(arr == threshold)

CSV_PATH = Path(__file__).resolve().parent / "common" / "insole_frames_flat.csv"
df = pd.read_csv(CSV_PATH)

# Separate left (client 0) and right (client 1)
left_df  = df[df["client"] == 0].reset_index(drop=True)
right_df = df[df["client"] == 1].reset_index(drop=True)

# Extract timestamps
left_t  = left_df["timestamp"].to_numpy()
right_t = right_df["timestamp"].to_numpy()

# Extract flattened 33x15 frames
left_vals  = left_df.filter(like="ch_").to_numpy(dtype=np.float32)
right_vals = right_df.filter(like="ch_").to_numpy(dtype=np.float32)

# Stop if there is no frame
if len(left_t) == 0 or len(right_t) == 0:
    raise SystemExit("Missing left or right frames")

# For each left_t, find closest right_t
idx = np.searchsorted(right_t, left_t, side="left")

idx_before = idx - 1
idx_after  = idx

valid_before = (idx_before >= 0)
valid_after  = (idx_after < len(right_t))

dt_before = np.full(left_t.shape, np.inf, dtype=np.float64)
dt_after  = np.full(left_t.shape, np.inf, dtype=np.float64)

dt_before[valid_before] = np.abs(right_t[idx_before[valid_before]] - left_t[valid_before])
dt_after[valid_after]   = np.abs(right_t[idx_after[valid_after]]   - left_t[valid_after])

pick_after = dt_after < dt_before
best_idx = np.where(pick_after, idx_after, idx_before)
best_dt  = np.minimum(dt_before, dt_after)

# Keep only pairs within MAX_DELTA
MAX_DELTA = 0.02
keep = best_dt <= MAX_DELTA

paired_left  = left_vals[keep]
paired_right = right_vals[best_idx[keep]]
paired_avg_timestamps = (left_t[keep] + right_t[best_idx[keep]]) / 2.0
paired_avg_timestamps = np.round(paired_avg_timestamps * 1000).astype(np.int64)

paired_avg_timestamps_str = np.array(
    [f"{ts:08d}" for ts in paired_avg_timestamps]
)

print("Final paired shapes:", paired_left.shape, paired_right.shape)
print("Average timestamp shape:", paired_avg_timestamps.shape)

# ---------- Rebuild 2D matrices directly from flat 33x15 ----------
rows, cols = 33, 15

left_mats  = paired_left.reshape(-1, rows, cols)
right_mats = paired_right.reshape(-1, rows, cols)

right_mats = right_mats[:, :, ::-1] # Flip right foot horizontally

print("2D shapes:", left_mats.shape, right_mats.shape)

# ---------- Normalize to [0, 1] ----------
MAX_R = 50000.0
left_norm  = np.clip(left_mats,  0.0, MAX_R) / MAX_R
right_norm = np.clip(right_mats, 0.0, MAX_R) / MAX_R

# ---------- Define inactive/faulty zones ----------
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

active_mask = (image_pattern == 1)
inactive_mask = ~active_mask

# Faulty sensors also considered inactive
inactive_mask[0][7] = True
inactive_mask[1][10] = True
inactive_mask[3][4] = True
inactive_mask[4][11] = True

# Put inactive/faulty zones to 0
left_norm[:, inactive_mask] = 0.0
right_norm[:, inactive_mask] = 0.0

# Invert sensor values: 1 = pressed, 0 = not pressed
left_norm  = 1.0 - left_norm
right_norm = 1.0 - right_norm

# Keep inactive/faulty zones at 0 after inversion
left_norm[:, inactive_mask] = 0.0
right_norm[:, inactive_mask] = 0.0

# ---------- Final ML format ----------
X = np.stack((left_norm, right_norm), axis=1)

print("Final X shape:", X.shape)

# ---------- Save NPZ ----------
npz_path = "data_for_3djoint"
OUTPUT_FILENAME = "sample_insole_dataset.npz"
os.makedirs(npz_path, exist_ok=True)
np.savez(
    os.path.join(npz_path, OUTPUT_FILENAME),
    X=X,
    avg_timestamps=paired_avg_timestamps,
)
