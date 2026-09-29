"""
Remove the camera tilt from a session's SAM-3D-Body joints so the subject walks
on a horizontal plane again.

WHY THIS EXISTS
---------------
The capture rig's camera was hand-placed, not levelled. SAM-3D-Body reports its
joints in the camera frame (with the mid-hip pinned to a canonical point), so a
tilted camera makes the whole lower limb travel along a sloped plane: in the
rendered GIFs the subject leans, and the "ground" wanders instead of staying
flat. `normalize_orientation()` in part2_preprocessing.ipynb only cancels YAW
(rotation about the vertical), so the roll and pitch survive untouched and land
straight in the training targets.

Measured tilt of the up-axis, before this script:

    ym                 36 deg
    s20260805_140850   25 deg
    yashanxiao         13 deg

That is a systematic, per-session rotation of every target the model has to
predict, and it is largest exactly where the errors are largest (heel/toe).

HOW THE TILT IS ESTIMATED
-------------------------
Two independent estimates of gravity in the camera frame, both reduced to "find
the unit vector u that is perpendicular to a set of things that should be
horizontal":

  plane  The ground. Plantar pressure says when a foot is flat on the floor
         (forefoot AND rearfoot both loaded), and the heel and big-toe of that
         foot are then ground contact points. Every such point should sit at the
         same height, so a robust total-least-squares plane through them has
         gravity as its normal. Pelvis bob (a few cm) is the noise floor.

  hips   The pelvis. The left-hip -> right-hip line is horizontal on average, so
         u is the direction most perpendicular to every hip vector in the
         session: the smallest eigenvector of sum(h h^T). This only works
         because the subject turns around and walks back, which makes the hip
         vectors span the whole horizontal plane -- the printed eigen-gap is the
         check that they did.

They agree to 1-4 deg on all three IMU sessions, which is the reason to trust
either. `--source both` averages them; disagreement is always printed.

The IMU is deliberately NOT used here. Its accelerometer measures gravity in the
FOOT frame, and recovering the camera tilt from that needs the foot's orientation
in the camera frame -- which is the unknown we are solving for. Pressure gives
the same stance information without the circularity.

WHAT IS WRITTEN
---------------
`data/3djoints_levelled/<session>_joints.npz`, byte-compatible with
`data/3djoints/` (joints, timestamps_ms, joint_names) so part2_preprocessing.ipynb
consumes it unchanged, plus the estimated `up_vector` and `tilt_deg` for audit.

One rotation per session (`--mode constant`) is the validated default. The camera
did also drift within a session, but only by 3-8 deg, and re-estimating in 30 s
windows (`--mode smooth`) did not improve either flatness metric on any of the
three sessions -- the windowed fits are noisier than the drift they chase.

USAGE
    python insole_collection/level_ground_joints.py --all
    python insole_collection/level_ground_joints.py ym --report
    python insole_collection/level_ground_joints.py ym --mode smooth
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
JOINTS_IN = REPO / "data" / "3djoints"
INSOLES = REPO / "data" / "insoles"
JOINTS_OUT = REPO / "data" / "3djoints_levelled"

# Y is down-positive in the SAM-3D-Body frame (mid-hip sits at y ~ -0.90, the
# heels at ~ -0.25), so "up" is -Y.
UP = np.array([0.0, -1.0, 0.0])

MAX_DELTA = 150.0     # ms, same pressure/joint pairing bound as part2_preprocessing
CONTACT_THR = 0.35    # normalised load above which a foot region counts as loaded
SMOOTH_S = 30.0       # window for --mode smooth
MAX_WINDOW_DEV = 15.0 # deg, reject a window fit further than this from the session mean

# Same array as part2_preprocessing.ipynb, cell "FOOT MASKS". Row 0 is the toe
# end (the mask widens at rows 8-9, the metatarsal heads) and row 32 the heel.
LEFT_FOOT_MASK_RAW = np.array([
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
], dtype=bool)
RIGHT_FOOT_MASK_RAW = np.fliplr(LEFT_FOOT_MASK_RAW)

FOREFOOT_ROWS = slice(0, 13)
REARFOOT_ROWS = slice(22, 33)


# ---------------------------------------------------------------- stance from pressure

def _nearest(source_t, target_t):
    """Index of the nearest source sample for each target, and the gap in ms."""
    idx = np.clip(np.searchsorted(source_t, target_t), 1, len(source_t) - 1)
    prev = idx - 1
    take_prev = np.abs(source_t[prev] - target_t) <= np.abs(source_t[idx] - target_t)
    best = np.where(take_prev, prev, idx)
    return best, np.abs(source_t[best] - target_t)


def _norm01(v, lo=5, hi=95):
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    return np.clip((v - a) / max(b - a, 1e-9), 0.0, 1.0)


def region_loads(X):
    """
    (M, 2, 33, 15) raw pressure -> {foot: (forefoot_load, rearfoot_load)}, each
    normalised to [0, 1] within the session.

    The right insole on this rig has ~150-200 sensors stuck at a large constant
    offset, so its total never returns to zero during swing. Subtracting a
    per-sensor 5th-percentile baseline restores the modulation; normalising per
    foot afterwards means one bad insole cannot set the threshold for the other.
    """
    loads = {}
    for foot, mask in ((0, LEFT_FOOT_MASK_RAW), (1, RIGHT_FOOT_MASK_RAW)):
        Xf = X[:, foot].astype(np.float64)
        baseline = np.percentile(Xf, 5, axis=0)
        Xc = np.clip(Xf - baseline, 0.0, None) * mask
        loads[foot] = (_norm01(Xc[:, FOREFOOT_ROWS].sum(axis=(1, 2))),
                       _norm01(Xc[:, REARFOOT_ROWS].sum(axis=(1, 2))))
    return loads


def flat_stance_mask(X, insole_t, joint_t, thr=CONTACT_THR):
    """Per joint frame, is each foot flat on the floor? -> (N, 2) bool."""
    idx, dt = _nearest(insole_t, joint_t)
    in_time = dt <= MAX_DELTA
    loads = region_loads(X)
    flat = np.zeros((len(joint_t), 2), dtype=bool)
    for foot in (0, 1):
        fore, rear = loads[foot]
        flat[:, foot] = in_time & (fore[idx] > thr) & (rear[idx] > thr)
    return flat


# ---------------------------------------------------------------- gravity estimators

def _cauchy_reweight(residual):
    sigma = 1.4826 * np.median(residual) + 1e-9
    return 1.0 / (1.0 + (residual / (2.0 * sigma)) ** 2)


def up_from_plane(points, n_iter=8):
    """Robust TLS plane through ground-contact points; its normal is gravity."""
    if len(points) < 50:
        return None, np.inf
    w = np.ones(len(points))
    for _ in range(n_iter):
        centre = (w[:, None] * points).sum(0) / w.sum()
        _, sv, Vt = np.linalg.svd((points - centre) * np.sqrt(w)[:, None],
                                  full_matrices=False)
        u = Vt[-1]
        w = _cauchy_reweight(np.abs((points - centre) @ u))
    # thickness of the fitted slab relative to its extent: <<1 means a real plane
    return u * np.sign(u @ UP), float(sv[-1] / max(sv[0], 1e-9))


def up_from_hips(hip_vectors, n_iter=8):
    """
    Gravity = the direction most perpendicular to every hip vector.

    Returns (u, gap) where gap is eigenvalue[1]/eigenvalue[0]. A large gap means
    the hip vectors really did span the horizontal plane, i.e. the subject walked
    in more than one direction; a gap near 1 means the estimate is degenerate.
    """
    H = hip_vectors / (np.linalg.norm(hip_vectors, axis=1, keepdims=True) + 1e-9)
    if len(H) < 50:
        return None, 0.0
    w = np.ones(len(H))
    for _ in range(n_iter):
        vals, vecs = np.linalg.eigh((H * w[:, None]).T @ H)
        u = vecs[:, 0]
        w = _cauchy_reweight(np.abs(H @ u))
    return u * np.sign(u @ UP), float(vals[1] / max(vals[0], 1e-9))


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def rotation_to_up(u):
    """Smallest rotation carrying u onto UP -- no yaw is introduced."""
    u = u / np.linalg.norm(u)
    axis = np.cross(u, UP)
    s, c = np.linalg.norm(axis), float(u @ UP)
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


# ---------------------------------------------------------------- report metrics

def flatness_report(J, names, flat, tag):
    """How level is the walking surface, and how level is the pelvis?"""
    heights, hip_tilts = [], []
    for foot, side in ((0, "left"), (1, "right")):
        sel = flat[:, foot]
        if sel.sum() < 20:
            continue
        for jn in (f"{side}-heel", f"{side}-big-toe-tip"):
            heights.append((J[sel, names.index(jn)] @ UP))
    lh, rh = names.index("left-hip"), names.index("right-hip")
    hv = J[:, rh] - J[:, lh]
    hip_tilts = np.degrees(np.arcsin(np.clip(
        (hv @ UP) / (np.linalg.norm(hv, axis=1) + 1e-9), -1, 1)))
    h = np.concatenate(heights) if heights else np.array([np.nan])
    print(f"    {tag:9s} ground-contact height  std {np.std(h) * 100:5.1f} cm   "
          f"p5-p95 spread {np.ptp(np.percentile(h, [5, 95])) * 100:5.1f} cm")
    print(f"    {tag:9s} pelvis obliquity       median {np.median(hip_tilts):+6.2f} deg  "
          f"std {np.std(hip_tilts):5.2f} deg")


# ---------------------------------------------------------------- main per-session

def level_session(session, source, mode, write):
    jp = JOINTS_IN / f"{session}_joints.npz"
    ip = INSOLES / f"{session}_insoles.npz"
    if not jp.exists():
        print(f"[{session}] missing {jp}")
        return False
    if not ip.exists():
        print(f"[{session}] missing {ip}")
        return False

    jz = np.load(jp, allow_pickle=True)
    J = np.asarray(jz["joints"], dtype=np.float64)
    tj = np.asarray(jz["timestamps_ms"], dtype=np.float64)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in jz["joint_names"]]

    iz = np.load(ip)
    flat = flat_stance_mask(np.asarray(iz["X"]), np.asarray(iz["avg_timestamps"],
                                                            dtype=np.float64), tj)

    lh, rh = names.index("left-hip"), names.index("right-hip")
    hip_vec = J[:, rh] - J[:, lh]
    ground_idx = [names.index(f"{s}-{p}") for s in ("left", "right")
                  for p in ("heel", "big-toe-tip")]

    def ground_points(rows):
        out = []
        for foot, side in ((0, "left"), (1, "right")):
            sel = rows & flat[:, foot]
            for jn in (f"{side}-heel", f"{side}-big-toe-tip"):
                out.append(J[sel, names.index(jn)])
        return np.concatenate(out) if out else np.zeros((0, 3))

    all_rows = np.ones(len(tj), dtype=bool)
    u_plane, slab = up_from_plane(ground_points(all_rows))
    u_hips, gap = up_from_hips(hip_vec)

    print(f"[{session}] {J.shape[0]} frames, {tj[-1] / 1000:.0f} s | "
          f"flat stance L={flat[:, 0].sum()} R={flat[:, 1].sum()}")
    if u_plane is not None:
        print(f"[{session}]   plane  up={np.round(u_plane, 4)}  "
              f"tilt {angle_deg(u_plane, UP):5.2f} deg  slab/extent {slab:.3f}")
    if u_hips is not None:
        print(f"[{session}]   hips   up={np.round(u_hips, 4)}  "
              f"tilt {angle_deg(u_hips, UP):5.2f} deg  eigen-gap {gap:.0f}x"
              f"{'  <-- DEGENERATE, subject never turned' if gap < 5 else ''}")
    if u_plane is not None and u_hips is not None:
        print(f"[{session}]   estimators disagree by {angle_deg(u_plane, u_hips):.2f} deg")

    picked = {"plane": u_plane, "hips": u_hips}.get(source)
    if source == "both":
        parts = [v for v in (u_plane, u_hips) if v is not None]
        picked = None if not parts else sum(parts) / np.linalg.norm(sum(parts))
    if picked is None:
        print(f"[{session}]   cannot estimate gravity with source={source} - skipped")
        return False

    print(f"[{session}]   using '{source}' -> tilt {angle_deg(picked, UP):.2f} deg")

    # Rotate about each frame's own mid-hip so the canonical root pin is kept.
    mid = 0.5 * (J[:, lh] + J[:, rh])
    if mode == "constant":
        R = rotation_to_up(picked)
        Jc = (J - mid[:, None, :]) @ R.T + mid[:, None, :]
        up_track = np.tile(picked, (len(tj), 1))
    else:
        # The camera drifted a few degrees over ym's two minutes. Re-estimate in
        # overlapping windows, then smooth hard: a per-frame fit would start
        # absorbing real pelvic obliquity instead of camera motion.
        fps = 1000.0 / max(np.median(np.diff(tj)), 1e-9)
        win = max(int(SMOOTH_S * fps), 60)
        centres, ups = [], []
        for s in range(0, max(len(tj) - win, 1), max(win // 4, 1)):
            rows = np.zeros(len(tj), dtype=bool)
            rows[s:s + win] = True
            uw_p, _ = up_from_plane(ground_points(rows))
            uw_h, gw = up_from_hips(hip_vec[s:s + win])
            if gw < 5:
                uw_h = None            # this window has no turn; hips say nothing
            parts = [v for v in (uw_p, uw_h) if v is not None]
            if not parts:
                continue
            uw = sum(parts) / np.linalg.norm(sum(parts))
            # A window with too little turning or too few stance frames can throw
            # a wild fit; on ym one window came out 43 deg off. The camera never
            # moves that far, so treat it as a failed fit rather than motion.
            if angle_deg(uw, picked) > MAX_WINDOW_DEV:
                continue
            centres.append(s + win // 2)
            ups.append(uw)
        if len(ups) < 3:
            print(f"[{session}]   too few usable windows - falling back to constant")
            return level_session(session, source, "constant", write)
        ups = np.array(ups)
        up_track = np.stack([np.interp(np.arange(len(tj)), centres, ups[:, k])
                             for k in range(3)], axis=1)
        up_track /= np.linalg.norm(up_track, axis=1, keepdims=True)
        drift = max(angle_deg(u, picked) for u in ups)
        print(f"[{session}]   smooth mode: {len(ups)} windows, "
              f"max deviation from the session mean {drift:.2f} deg")
        Jc = np.empty_like(J)
        for i in range(len(tj)):
            R = rotation_to_up(up_track[i])
            Jc[i] = (J[i] - mid[i]) @ R.T + mid[i]

    print(f"[{session}]   before/after:")
    flatness_report(J, names, flat, "before")
    flatness_report(Jc, names, flat, "after")

    if not write:
        print(f"[{session}]   report only, nothing written\n")
        return True

    JOINTS_OUT.mkdir(parents=True, exist_ok=True)
    out = JOINTS_OUT / f"{session}_joints.npz"
    np.savez_compressed(
        out,
        joints=Jc.astype(np.float32),
        timestamps_ms=tj,
        joint_names=jz["joint_names"],
        up_vector=up_track.astype(np.float32),
        tilt_deg=np.float32(angle_deg(picked, UP)),
        flat_stance=flat,
        levelling_source=np.array(source),
        levelling_mode=np.array(mode),
    )
    print(f"[{session}]   -> {out}\n")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--all", action="store_true",
                    help="every session that has both joints and insoles")
    ap.add_argument("--source", choices=["both", "plane", "hips"], default="both")
    ap.add_argument("--mode", choices=["constant", "smooth"], default="constant")
    ap.add_argument("--report", action="store_true", help="print without writing")
    args = ap.parse_args()

    names = args.sessions
    if args.all:
        names = sorted(p.stem.replace("_joints", "") for p in JOINTS_IN.glob("*_joints.npz")
                       if (INSOLES / f"{p.stem.replace('_joints', '')}_insoles.npz").exists())
    if not names:
        ap.error("give at least one session name, or --all")

    ok = sum(level_session(n, args.source, args.mode, write=not args.report) for n in names)
    print(f"Levelled {ok}/{len(names)} session(s) -> {JOINTS_OUT}")


if __name__ == "__main__":
    sys.exit(main())
