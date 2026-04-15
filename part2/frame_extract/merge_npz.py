import numpy as np
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
SESSION_NAME = "brahim1"

JOINTS_FOLDER = Path(f"outputs/joints_npz/{SESSION_NAME}")
OUTPUT_PATH   = Path(f"outputs/sessions_final/{SESSION_NAME}_joints.npz")
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

npz_files = sorted(JOINTS_FOLDER.glob("*.npz"), key=lambda p: int(p.stem))

if not npz_files:
    raise SystemExit(f"No NPZ files found in {JOINTS_FOLDER}")

first_valid = None
for f in npz_files:
    data = np.load(f, allow_pickle=True)
    if "person_0" in data:
        first_valid = data
        break

if first_valid is None:
    raise SystemExit("No file contains person_0")

print(f"Keys in first valid NPZ: {list(first_valid.keys())}")
for k, v in first_valid.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

joint_names = first_valid["joint_names"]

joints_list = []
timestamps_ms = []
missing_person0 = []

prev_points = None

for npz_file in npz_files:
    data = np.load(npz_file, allow_pickle=True)

    if "person_0" in data:
        points = data["person_0"]
        if points.ndim == 3:
            points = points[0]
        points = np.asarray(points, dtype=np.float32)
        prev_points = points.copy()
    else:
        missing_person0.append(npz_file.name)

        if prev_points is None:
            continue

        points = prev_points.copy()

    joints_list.append(points)
    timestamps_ms.append(float(npz_file.stem))

if not joints_list:
    raise SystemExit("No valid joints found to merge")

joints = np.stack(joints_list, axis=0)
timestamps_ms = np.array(timestamps_ms, dtype=np.float64)

print(f"\nSession: {SESSION_NAME}")
print(f"joints: {joints.shape} | timestamps: {timestamps_ms.shape} | duration: {timestamps_ms[-1] / 1000:.3f}s")
print(f"missing person_0: {len(missing_person0)}")
if missing_person0:
    print("npz without person_0:", ", ".join(missing_person0))

np.savez_compressed(
    OUTPUT_PATH,
    joints=joints,
    timestamps_ms=timestamps_ms,
    joint_names=joint_names,
)

print(f"Saved to {OUTPUT_PATH}")