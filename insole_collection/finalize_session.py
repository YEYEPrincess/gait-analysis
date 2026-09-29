"""
Rebase a session's pressure AND IMU onto the video's 0-based clock, then stage
them where part2_preprocessing.ipynb expects.

WHY THIS EXISTS
---------------
offline_processing_for_3djoint.py and offline_processing_imu.py both store Unix
epoch milliseconds. The joint stream, in contrast, starts at 0 because
frame_calibration.py renamed the frames to milliseconds-from-session-start. So
before anything can be synced, the two insole-side streams have to be shifted
onto that same 0-based clock.

`visualise_for_3djoint.py` does this for pressure only, interactively. Doing the
IMU separately by hand invites the failure this script exists to prevent: if the
two streams are rebased against different origins they end up silently offset
from each other, which is exactly the class of bug that cost this project a
15/16 frame-rate error earlier on.

Here both streams are shifted by one shared origin, so they cannot drift apart.

CHOOSING THE TRIM
-----------------
`--trim-s` is how much of the START of the insole/IMU recording to discard,
i.e. how much earlier the insole started than the video. Get it from:

    python joint_collection/joint_processing/suggest_start_frame.py <session>

A negative "best lag" there means the insole leads by that much; pass its
absolute value here. A lag near zero means the video was already trimmed to
match, so pass 0.

USAGE
    python insole_collection/finalize_session.py s20260805_140850 --trim-s 0
    python insole_collection/finalize_session.py yashanxiao --trim-s 0.67
    python insole_collection/finalize_session.py ym --trim-s 0 --dry-run
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "insole_collection" / "data_for_3djoint"
OUT_PRESSURE = REPO / "data" / "insoles"
OUT_IMU = REPO / "data" / "imu"
JOINTS = REPO / "data" / "3djoints"


def load(path, key):
    d = np.load(path, allow_pickle=True)
    return np.asarray(d[key]), np.asarray(d["avg_timestamps"], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--trim-s", type=float, default=0.0,
                    help="seconds to drop from the start of the insole/IMU streams")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    s = args.session

    p_press = SRC / f"{s}_insole.npz"
    p_imu = SRC / f"{s}_imu.npz"
    if not p_press.exists():
        raise SystemExit(f"Missing {p_press}. Run offline_processing_for_3djoint.py first.")

    X, t_press = load(p_press, "X")
    have_imu = p_imu.exists()
    if have_imu:
        imu, t_imu = load(p_imu, "imu")
    else:
        print(f"[{s}] no IMU npz - staging pressure only")

    # One origin for both streams: the pressure clock's start plus the trim.
    origin = t_press[0] + args.trim_s * 1000.0
    print(f"[{s}] trim {args.trim_s:+.2f}s -> shared origin {origin:.0f} ms (epoch)")

    keep_p = t_press >= origin
    Xc, tpc = X[keep_p], t_press[keep_p] - origin
    print(f"[{s}] pressure {X.shape} -> {Xc.shape}, "
          f"{tpc[-1]/1000:.1f}s, dt median {np.median(np.diff(tpc)):.0f} ms "
          f"(dropped {int((~keep_p).sum())} frames)")

    if have_imu:
        keep_i = t_imu >= origin
        imuc, tic = imu[keep_i], t_imu[keep_i] - origin
        print(f"[{s}] imu      {imu.shape} -> {imuc.shape}, "
              f"{tic[-1]/1000:.1f}s, dt median {np.median(np.diff(tic)):.0f} ms "
              f"(dropped {int((~keep_i).sum())} frames)")
        # Both were shifted by the same `origin`, so any residual offset here is
        # in the capture itself, not introduced by this step.
        print(f"[{s}] start offset after rebase: pressure {tpc[0]:.0f} ms, imu {tic[0]:.0f} ms")

    jp = JOINTS / f"{s}_joints.npz"
    if jp.exists():
        tj = np.asarray(np.load(jp, allow_pickle=True)["timestamps_ms"], dtype=np.float64)
        print(f"[{s}] joints   {tj[-1]/1000:.1f}s, dt median {np.median(np.diff(tj)):.0f} ms")
        overlap = min(tpc[-1], tj[-1]) / 1000
        print(f"[{s}] usable overlap with the joint stream: {overlap:.1f}s")
    else:
        print(f"[{s}] no {jp.name} yet - merge the SAM output first")

    if args.dry_run:
        print(f"[{s}] dry run, nothing written")
        return

    OUT_PRESSURE.mkdir(parents=True, exist_ok=True)
    out_p = OUT_PRESSURE / f"{s}_insoles.npz"
    np.savez_compressed(out_p, X=Xc, avg_timestamps=tpc)
    print(f"[{s}] -> {out_p}")

    if have_imu:
        OUT_IMU.mkdir(parents=True, exist_ok=True)
        out_i = OUT_IMU / f"{s}_imu.npz"
        np.savez_compressed(out_i, imu=imuc, avg_timestamps=tic,
                            axes=np.array(["ax", "ay", "az", "gx", "gy", "gz"], dtype=object))
        print(f"[{s}] -> {out_i}")

    print(f"\n[{s}] ready for part2_preprocessing.ipynb")


if __name__ == "__main__":
    sys.exit(main())
