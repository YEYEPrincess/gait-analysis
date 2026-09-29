"""
Merge the per-frame joint .npz files written by joint_extract.py into one file
per session, choosing which detected person is the subject.

WHY THE CHOICE MATTERS
----------------------
joint_extract.py stores every detected person as person_0, person_1, ... in the
order the detector returned them, and that order comes from

    np.lexsort((boxes[:,3], boxes[:,2], boxes[:,1], boxes[:,0]))

i.e. sorted by the box's left edge. So person_0 is simply the left-most person in
the frame, not the subject. In the ym session a colleague is seated at the left
of the frame, so person_0 tracked the colleague: 48% of that session's poses came
out non-upright (feet above the hip) because a seated, half-occluded person was
being fitted.

SELECTION MODES
    track    (default) largest box, then stay with whichever box moves least
             frame to frame. Robust when a bystander drifts in and out.
    largest  largest box area each frame, no temporal memory.
    first    person_0, i.e. the old behaviour. Kept so results can be reproduced.

Timestamps come from the frame filenames, which frame_calibration.py already
rewrote as milliseconds from the session start.

Usage:
    python merge_npz.py --all
    python merge_npz.py ym --select track
    python merge_npz.py ym --report          # inspect without writing
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JOINTS_ROOT = HERE / "outputs" / "joints_npz"
OUTPUT_ROOT = HERE / "outputs" / "sessions_final"

# How strongly to prefer a box near the previous frame's pick over a bigger one.
# 1.0 means a full-diagonal jump costs as much as losing the entire area score.
TRACK_PENALTY = 1.0


def person_keys(data):
    keys = [k for k in data.files if k.startswith("person_")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def choose(data, mode, prev_center):
    """Return (points, index, centre) for the chosen person, or (None, -1, prev)."""
    keys = person_keys(data)
    if not keys:
        return None, -1, prev_center

    boxes = np.asarray(data["bboxes"], dtype=np.float64) if "bboxes" in data.files else None
    if mode == "first" or boxes is None or len(boxes) != len(keys):
        idx = 0
    else:
        wh = np.clip(boxes[:, 2:4] - boxes[:, 0:2], 0, None)
        areas = wh[:, 0] * wh[:, 1]
        centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2.0
        area_score = areas / (areas.max() + 1e-9)

        if mode == "largest" or prev_center is None:
            idx = int(np.argmax(area_score))
        else:
            diag = float(np.hypot(*(boxes[:, 2:4].max(axis=0) - boxes[:, 0:2].min(axis=0))))
            dist = np.linalg.norm(centers - prev_center, axis=1) / (diag + 1e-9)
            idx = int(np.argmax(area_score - TRACK_PENALTY * dist))

    points = np.asarray(data[keys[idx]], dtype=np.float32)
    if points.ndim == 3:
        points = points[0]
    center = None
    if boxes is not None and idx < len(boxes):
        center = (boxes[idx, 0:2] + boxes[idx, 2:4]) / 2.0
    return points, idx, center if center is not None else prev_center


def merge(session: str, mode: str, write: bool) -> bool:
    folder = JOINTS_ROOT / session
    if not folder.is_dir():
        print(f"[{session}] no folder at {folder}")
        return False

    npz_files = sorted(folder.glob("*.npz"), key=lambda p: int(p.stem))
    if not npz_files:
        print(f"[{session}] no .npz files in {folder}")
        return False

    joint_names = None
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        if person_keys(d):
            joint_names = d["joint_names"]
            break
    if joint_names is None:
        print(f"[{session}] no frame contains a person")
        return False

    joints_list, timestamps_ms = [], []
    prev_points, prev_center = None, None
    n_missing = n_multi = n_not_first = 0

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        keys = person_keys(data)
        if len(keys) > 1:
            n_multi += 1

        points, idx, prev_center = choose(data, mode, prev_center)
        if points is None:
            n_missing += 1
            if prev_points is None:
                continue                     # nothing to carry forward yet
            points = prev_points.copy()
        else:
            if idx != 0:
                n_not_first += 1
            prev_points = points.copy()

        joints_list.append(points)
        timestamps_ms.append(float(npz_file.stem))

    if not joints_list:
        print(f"[{session}] no valid joints to merge")
        return False

    joints = np.stack(joints_list, axis=0)
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64)

    # Upright = vertical share of the hip->toe vector; a seated or badly fitted
    # person scores low, so this is the quickest read on whether the choice worked.
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in joint_names]
    upright = np.nan
    if "left-hip" in names and "left-big-toe-tip" in names:
        v = joints[:, names.index("left-big-toe-tip")] - joints[:, names.index("left-hip")]
        upright = float(np.mean(v[:, 1] / (np.linalg.norm(v, axis=1) + 1e-9)))

    n = len(npz_files)
    print(f"[{session}] {joints.shape} | {timestamps_ms[-1]/1000:.1f}s | "
          f"dt {np.median(np.diff(timestamps_ms)):.0f} ms")
    print(f"[{session}]   multi-person frames {n_multi}/{n} ({100*n_multi/n:.1f}%) | "
          f"picked someone other than person_0 in {n_not_first} | "
          f"no detection {n_missing}")
    print(f"[{session}]   mean upright {upright:.3f}   "
          f"({'OK' if upright > 0.85 else 'LOW - the chosen person may still be wrong'})")

    if not write:
        print(f"[{session}]   report only, nothing written")
        return True

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / f"{session}_joints.npz"
    np.savez_compressed(out_path, joints=joints, timestamps_ms=timestamps_ms,
                        joint_names=joint_names)
    print(f"[{session}] -> {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--select", choices=["track", "largest", "first"], default="track")
    ap.add_argument("--report", action="store_true", help="print stats without writing")
    args = ap.parse_args()

    names = ([p.name for p in sorted(JOINTS_ROOT.iterdir()) if p.is_dir()]
             if args.all else args.sessions)
    if not names:
        ap.error("give at least one session name, or --all")

    print(f"selection mode: {args.select}\n")
    ok = sum(merge(n, args.select, write=not args.report) for n in names)
    print(f"\nProcessed {ok}/{len(names)} session(s).")


if __name__ == "__main__":
    sys.exit(main())
