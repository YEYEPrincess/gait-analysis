"""
Turn a raw IMU capture into the (M, 2, 6) array the models consume.

The collector writes one row per foot per sample:

    timestamp, foot_side, ax, ay, az, gx, gy, gz

Left (foot_side 0) and right (foot_side 1) arrive interleaved on separate rows,
so they are paired by nearest timestamp exactly the way
`offline_processing_for_3djoint.py` pairs the two insoles. The result matches
BALANCE's layout (`APP_Fall_Risk_Prediction/Dataset/data_preprocessing.py`),
which stacks `IMU_L` and `IMU_R` into `(T, 2, 6)`.

Units: the ICM-45686 is logged as raw int16 counts. Accelerometer counts are
converted to g (1 g ~ 8192 LSB at the +/-4 g range this rig uses, checked against
the ~8200 count magnitude of a stationary foot) and gyro counts to a comparable
scale, because the two channels otherwise differ by ~2 orders of magnitude and
whichever is larger dominates the first layer. BALANCE feeds raw counts straight
into `nn.Linear`; normalising here is the one deliberate difference.

Usage:
    python insole_collection/offline_processing_imu.py wuyu
    python insole_collection/offline_processing_imu.py wuyu --in path/to/imu.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Where the collector drops raw captures, and where the pressure script writes.
RECORDINGS_DIRS = [
    Path(r"d:\Desktop\UCL\Msc project\BALANCE\HW_ESP32_ADC_Current_driver_active"
         r"\DataCollection\recordings"),
    Path(__file__).resolve().parent / "common",
    Path(r"d:\Desktop\UCL\Msc project\v1_Smart_insole_unet"),
]
OUT_DIR = Path(__file__).resolve().parent / "data_for_3djoint"

MAX_DELTA = 0.02          # s, same left/right pairing tolerance as the pressure script
ACCEL_LSB_PER_G = 8192.0  # +/-4 g range
GYRO_LSB_PER_DPS = 16.4   # +/-2000 dps range

AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
FOOT_COL_ALIASES = ("foot_side", "client", "foot", "side")


def find_input(session: str) -> Path:
    patterns = [f"{session}imu*.csv", f"{session}_imu*.csv", f"imu_{session}*.csv",
                f"{session}.csv"]
    for root in RECORDINGS_DIRS:
        if not root.exists():
            continue
        for pat in patterns:
            hits = sorted(root.glob(pat))
            if hits:
                return hits[-1]        # newest capture wins
    raise SystemExit(
        f"No IMU CSV found for '{session}'. Looked for {patterns} under:\n  "
        + "\n  ".join(str(r) for r in RECORDINGS_DIRS)
        + "\nPass an explicit path with --in."
    )


def pair_feet(t_left, v_left, t_right, v_right):
    """Nearest-timestamp pairing, identical in spirit to the pressure pipeline."""
    idx = np.searchsorted(t_right, t_left, side="left")
    before, after = idx - 1, idx

    dt_before = np.full(t_left.shape, np.inf)
    dt_after = np.full(t_left.shape, np.inf)
    ok_b, ok_a = before >= 0, after < len(t_right)
    dt_before[ok_b] = np.abs(t_right[before[ok_b]] - t_left[ok_b])
    dt_after[ok_a] = np.abs(t_right[after[ok_a]] - t_left[ok_a])

    best = np.where(dt_after < dt_before, after, before)
    keep = np.minimum(dt_before, dt_after) <= MAX_DELTA

    paired = np.stack([v_left[keep], v_right[best[keep]]], axis=1)   # (M, 2, 6)
    stamps = (t_left[keep] + t_right[best[keep]]) / 2.0
    return paired, stamps, keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session")
    parser.add_argument("--in", dest="src", default=None)
    parser.add_argument("--out", dest="dst", default=None)
    parser.add_argument("--raw-counts", action="store_true",
                        help="skip unit conversion and keep raw int16 counts (BALANCE behaviour)")
    args = parser.parse_args()

    src = Path(args.src) if args.src else find_input(args.session)
    dst = Path(args.dst) if args.dst else OUT_DIR / f"{args.session}_imu.npz"

    print(f"Reading  {src}")
    df = pd.read_csv(src)

    foot_col = next((c for c in FOOT_COL_ALIASES if c in df.columns), None)
    if foot_col is None:
        raise SystemExit(f"No foot column (tried {FOOT_COL_ALIASES}); columns: {list(df.columns)[:8]}")
    missing = [a for a in AXES if a not in df.columns]
    if missing:
        raise SystemExit(f"Missing IMU axes {missing}; columns: {list(df.columns)[:10]}")

    left = df[df[foot_col] == 0].reset_index(drop=True)
    right = df[df[foot_col] == 1].reset_index(drop=True)
    if len(left) == 0 or len(right) == 0:
        raise SystemExit(f"Need both feet, got left={len(left)} right={len(right)}")

    t_l = left["timestamp"].to_numpy(dtype=np.float64)
    t_r = right["timestamp"].to_numpy(dtype=np.float64)
    v_l = left[AXES].to_numpy(dtype=np.float32)
    v_r = right[AXES].to_numpy(dtype=np.float32)

    imu, stamps, keep = pair_feet(t_l, v_l, t_r, v_r)
    print(f"  left {len(t_l)} / right {len(t_r)} rows -> paired {len(imu)} "
          f"(dropped {int((~keep).sum())} outside {MAX_DELTA*1000:.0f} ms)")

    if not args.raw_counts:
        imu[:, :, 0:3] /= ACCEL_LSB_PER_G       # -> g
        imu[:, :, 3:6] /= GYRO_LSB_PER_DPS      # -> deg/s
        unit = "accel in g, gyro in deg/s"
    else:
        unit = "raw int16 counts"

    stamps_ms = np.round(stamps * 1000).astype(np.int64)

    a = np.linalg.norm(imu[:, :, 0:3], axis=2)
    g = np.linalg.norm(imu[:, :, 3:6], axis=2)
    print(f"  imu {imu.shape}  ({unit})")
    print(f"  |accel| median L/R = {np.median(a[:,0]):.2f} / {np.median(a[:,1]):.2f}")
    print(f"  |gyro|  p95    L/R = {np.percentile(g[:,0],95):.1f} / {np.percentile(g[:,1],95):.1f}")
    print(f"  duration {(stamps_ms[-1]-stamps_ms[0])/1000:.1f}s, "
          f"dt median {np.median(np.diff(stamps_ms)):.0f} ms")

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, imu=imu, avg_timestamps=stamps_ms,
             axes=np.array(AXES, dtype=object), units=np.array(unit))
    print(f"Saved -> {dst}")
    print(f"\nNext: trim/rebase it alongside the pressure npz, then copy to data/imu/{args.session}_imu.npz")


if __name__ == "__main__":
    sys.exit(main())
