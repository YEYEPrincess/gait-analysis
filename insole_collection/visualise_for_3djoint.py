import os
import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
FOOT_SELECTOR = "both"  # "left", "right", or "both"
JOINT_ESTIMATION = True
DATASET_NAME = "sample_insole_dataset"
START_IDX = 2000
END_IDX = 2200  # None = preview until the last frame
NPERCOL = 20
ENABLE_TRIM_AFTER_PREVIEW = False   # if True, ask user for first frame index to keep after preview

INDEX_FONT_SIZE = 7
TIMESTAMP_FONT_SIZE = 7
ROW_HEIGHT = 0.4
LABEL_Y = 0.96
PAIR_SPACER_WIDTH = 0.7

# ============================================================
# Masks
# ============================================================
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

inactive_mask = image_pattern == 0
inactive_mask[0, 7] = True
inactive_mask[1, 10] = True
inactive_mask[3, 4] = True
inactive_mask[4, 11] = True
inactive_mask_flipped = np.fliplr(inactive_mask)

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="lightgray")

# ============================================================
# Helpers
# ============================================================
def format_time_label(ts):
    if np.isclose(ts, round(ts)):
        return str(int(round(ts)))
    return f"{ts:.3f}"

def get_display_mode(foot_selector):
    foot_selector = foot_selector.lower()
    if foot_selector not in {"left", "right", "both"}:
        raise ValueError('FOOT_SELECTOR must be "left", "right", or "both".')
    return foot_selector


def create_frame_grid(display_mode, npercol, n_blocks, figsize):
    if display_mode == "both":
        ncol = max(1, 3 * n_blocks - 1)
        width_ratios = []
        for block in range(n_blocks):
            width_ratios.extend([1.0, 1.0])
            if block < n_blocks - 1:
                width_ratios.append(PAIR_SPACER_WIDTH)

        fig, axes = plt.subplots(
            npercol,
            ncol,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"width_ratios": width_ratios},
        )

        for spacer_col in range(2, ncol, 3):
            for row in range(npercol):
                axes[row, spacer_col].axis("off")

        return fig, axes

    fig, axes = plt.subplots(npercol, n_blocks, figsize=figsize, squeeze=False)
    return fig, axes


def show_frames_with_indices(X, foot_selector, npercol, start_idx=0, timestamps=None):
    n = X.shape[0]
    if n == 0:
        raise ValueError("No frames to display.")

    display_mode = get_display_mode(foot_selector)

    if display_mode == "both":
        n_blocks = math.ceil(n / npercol)
        figsize = (1.35 * n_blocks, ROW_HEIGHT * npercol)
    else:
        n_blocks = math.ceil(n / npercol)
        ncol = n_blocks
        figsize = (0.65 * ncol, ROW_HEIGHT * npercol)

    if display_mode != "both":
        ncol = n_blocks

    if display_mode == "both" and timestamps is not None and len(timestamps) != n:
        raise ValueError(f"Mismatch between X and timestamps: {n} vs {len(timestamps)}")

    fig, axes = create_frame_grid(display_mode, npercol, n_blocks, figsize)

    for ax in axes.flat:
        ax.set_aspect("auto")

    for i in range(n):
        row = i % npercol
        block = i // npercol
        index_label = f"{i + start_idx}"
        time_label = format_time_label(timestamps[i]) if timestamps is not None else index_label

        if display_mode == "both":
            left = X[i, 0].copy()
            right = X[i, 1].copy()

            left[inactive_mask] = np.nan
            right[inactive_mask_flipped] = np.nan

            col_left = 3 * block
            col_right = col_left + 1

            ax_l = axes[row, col_left]
            ax_r = axes[row, col_right]

            ax_l.imshow(left, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax_r.imshow(right, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

            ax_l.text(0.5, LABEL_Y, index_label, transform=ax_l.transAxes,
                      ha="center", va="bottom", fontsize=INDEX_FONT_SIZE)
            ax_r.text(0.5, LABEL_Y, time_label, transform=ax_r.transAxes,
                      ha="center", va="bottom", fontsize=TIMESTAMP_FONT_SIZE)

            ax_l.axis("off")
            ax_r.axis("off")
        else:
            if display_mode == "left":
                foot = X[i, 0].copy()
                foot[inactive_mask] = np.nan
            else:
                foot = X[i, 1].copy()
                foot[inactive_mask_flipped] = np.nan

            ax = axes[row, block]
            ax.imshow(foot, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.text(0.5, LABEL_Y, index_label, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=INDEX_FONT_SIZE)
            ax.axis("off")

    total_slots = npercol * n_blocks
    if display_mode == "both":
        for j in range(n, total_slots):
            row = j % npercol
            block = j // npercol
            axes[row, 3 * block].axis("off")
            axes[row, 3 * block + 1].axis("off")
    else:
        for j in range(n, total_slots):
            row = j % npercol
            block = j // npercol
            axes[row, block].axis("off")

    fig.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001, wspace=0.0, hspace=0.0)
    plt.show()

def show_frames_with_timestamps(X, timestamps, foot_selector, npercol):
    n = X.shape[0]
    if n == 0:
        raise ValueError("No frames to display.")

    display_mode = get_display_mode(foot_selector)

    if display_mode == "both":
        n_blocks = math.ceil(n / npercol)
        figsize = (1.35 * n_blocks, ROW_HEIGHT * npercol)
    else:
        n_blocks = math.ceil(n / npercol)
        ncol = n_blocks
        figsize = (0.65 * ncol, ROW_HEIGHT * npercol)

    if display_mode != "both":
        ncol = n_blocks

    fig, axes = create_frame_grid(display_mode, npercol, n_blocks, figsize)

    for ax in axes.flat:
        ax.set_aspect("auto")

    for i in range(n):
        row = i % npercol
        block = i // npercol
        label = format_time_label(timestamps[i])

        if display_mode == "both":
            left = X[i, 0].copy()
            right = X[i, 1].copy()

            left[inactive_mask] = np.nan
            right[inactive_mask_flipped] = np.nan

            col_left = 3 * block
            col_right = col_left + 1

            ax_l = axes[row, col_left]
            ax_r = axes[row, col_right]

            ax_l.imshow(left, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax_r.imshow(right, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

            ax_l.text(0.5, LABEL_Y, label, transform=ax_l.transAxes,
                      ha="center", va="bottom", fontsize=TIMESTAMP_FONT_SIZE)
            ax_r.text(0.5, LABEL_Y, label, transform=ax_r.transAxes,
                      ha="center", va="bottom", fontsize=TIMESTAMP_FONT_SIZE)

            ax_l.axis("off")
            ax_r.axis("off")
        else:
            if display_mode == "left":
                foot = X[i, 0].copy()
                foot[inactive_mask] = np.nan
            else:
                foot = X[i, 1].copy()
                foot[inactive_mask_flipped] = np.nan

            ax = axes[row, block]
            ax.imshow(foot, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.text(0.5, LABEL_Y, label, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=TIMESTAMP_FONT_SIZE)
            ax.axis("off")

    total_slots = npercol * n_blocks
    if display_mode == "both":
        for j in range(n, total_slots):
            row = j % npercol
            block = j // npercol
            axes[row, 3 * block].axis("off")
            axes[row, 3 * block + 1].axis("off")
    else:
        for j in range(n, total_slots):
            row = j % npercol
            block = j // npercol
            axes[row, block].axis("off")

    fig.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001, wspace=0.0, hspace=0.0)
    plt.show()

def ask_first_frame_to_keep(start_idx, end_idx):
    while True:
        user_input = input(
            f"Enter the first frame index to keep ({start_idx} to {end_idx}), or press Enter to keep all: "
        ).strip()

        if user_input == "":
            return 0

        try:
            absolute_idx = int(user_input)
            if start_idx <= absolute_idx <= end_idx:
                return absolute_idx - start_idx
            print(f"Please enter an integer between {start_idx} and {end_idx}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# ============================================================
# Load data
# ============================================================
npz_dir = "data_for_3djoint" if JOINT_ESTIMATION else "data_ct"
npz_path = os.path.join(npz_dir, f"{DATASET_NAME}.npz")
data = np.load(npz_path)
npz_stem, npz_ext = os.path.splitext(npz_path)

X = data["X"]

if "avg_timestamps" not in data:
    raise ValueError(f"{os.path.basename(npz_path)} does not contain 'avg_timestamps'.")

avg_timestamps = data["avg_timestamps"].astype(np.float64)

if X.shape[0] != avg_timestamps.shape[0]:
    raise ValueError(f"Mismatch between X and avg_timestamps: {X.shape[0]} vs {avg_timestamps.shape[0]}")

if X.ndim != 4 or X.shape[1] < 2:
    raise ValueError(f"Expected X with shape (N, 2, 33, 15), got {X.shape}")

if START_IDX < 0 or START_IDX >= X.shape[0]:
    raise ValueError("No frames available with current START_IDX.")

X_from_start = X[START_IDX:]
timestamps_from_start = avg_timestamps[START_IDX:]

if END_IDX is not None:
    if END_IDX < START_IDX:
        raise ValueError("END_IDX must be greater than or equal to START_IDX.")
    end_exclusive = min(END_IDX + 1, X.shape[0])
else:
    end_exclusive = X.shape[0]

X_view = X[START_IDX:end_exclusive]
timestamps_view = avg_timestamps[START_IDX:end_exclusive]

if X_view.shape[0] == 0:
    raise ValueError("No frames available within the current START_IDX and END_IDX preview range.")

# ============================================================
# First preview: frame indices only
# ============================================================
show_frames_with_indices(
    X=X_view,
    foot_selector=FOOT_SELECTOR,
    npercol=NPERCOL,
    start_idx=START_IDX,
    timestamps=timestamps_view
)

# ============================================================
# Optional trim after preview: crop actual NPZ
# ============================================================
if ENABLE_TRIM_AFTER_PREVIEW:
    first_keep = ask_first_frame_to_keep(START_IDX, end_exclusive - 1)

    X_cropped = X_from_start[first_keep:]
    timestamps_cropped = timestamps_from_start[first_keep:]

    if X_cropped.shape[0] == 0:
        raise ValueError("No frames remain after trimming.")

    timestamps_cropped = timestamps_cropped - timestamps_cropped[0]

    save_path = f"{npz_stem}_cropped{npz_ext}"

    np.savez(
        save_path,
        X=X_cropped,
        avg_timestamps=timestamps_cropped
    )

    print(f"Cropped NPZ saved to: {save_path}")
    print(f"Kept {X_cropped.shape[0]} frames starting from original frame index {START_IDX + first_keep}")

    # Reload cropped file to prove we're displaying the actual saved NPZ
    cropped_data = np.load(save_path)
    X_saved = cropped_data["X"]
    timestamps_saved = cropped_data["avg_timestamps"]

    cropped_start_idx = START_IDX + first_keep
    if END_IDX is not None:
        trimmed_end_exclusive = max(0, end_exclusive - cropped_start_idx)
    else:
        trimmed_end_exclusive = X_saved.shape[0]

    X_saved_view = X_saved[:trimmed_end_exclusive]
    timestamps_saved_view = timestamps_saved[:trimmed_end_exclusive]

    print("Showing cropped frames with rebased timestamps...")

    show_frames_with_timestamps(
        X=X_saved_view,
        timestamps=timestamps_saved_view,
        foot_selector=FOOT_SELECTOR,
        npercol=NPERCOL
    )
