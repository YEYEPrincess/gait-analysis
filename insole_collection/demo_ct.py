"""
──────────────────────────────────────────────────────────────────────────────
Real-time Crosstalk-free Rsensor Heat-map
──────────────────────────────────────────────────────────────────────────────
"""
import os
from pathlib import Path
from common.logger import FlatCsvFrameLogger

CSV_PATH = Path(__file__).resolve().parent / "common" / "insole_frames_flat.csv"
frame_logger = FlatCsvFrameLogger(str(CSV_PATH), n_channels=253)

import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import torch

from models.unet import UNetSmall

MODEL_NAME = "unet100_5_sig.pt"

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/checkpoint" / MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetSmall().to(device)
state_dict = torch.load(str(MODEL_PATH), map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ── UDP ──────────────────────────────────────────────────────────────────────
UDP_IP = os.getenv("INSOLE_UDP_IP", "0.0.0.0")
UDP_PORT = int(os.getenv("INSOLE_UDP_PORT", "8999"))

Frame_ONE_SIZE = 253
Frame_TWO_SIZE = 253
IMU_ENABLE     = False
CLIENT_ID_SIZE = 1
IMU_DATA_SIZE  = 6
BUFFER_SIZE    = (Frame_ONE_SIZE + Frame_TWO_SIZE + CLIENT_ID_SIZE + IMU_DATA_SIZE) * 2

MATRIX_ROW    = 33
MATRIX_COLUMN = 15

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
])

mask = image_pattern.copy().astype(np.float32)
mask[0][7]  = 0.0
mask[1][10] = 0.0
mask[3][4]  = 0.0
mask[4][11] = 0.0
mask = (mask != 0.0).astype(np.float32)

SHOW_BOTH_FEET   = True
SINGLE_FOOT_SIDE = 'right'

RSENSOR_VMIN   = 0
RSENSOR_VMAX   = 50_000
UPSCALE_BLUR   = True
UPSCALE_FACTOR = 10
BLUR_SIGMA     = 1.5

lookup_table = [
    16,32,41,59,70,81,91,101,111,121,130,139,140,147,154,161,162,168,174,180,192,199,200,207,214,221,228,235,236,242,248,249,
    4,24,33,50,60,71,82,92,102,112,122,131,132,141,148,155,156,163,169,175,186,193,194,201,208,215,222,229,230,237,243,244,
    10,17,25,42,51,61,72,83,93,103,113,123,124,133,142,149,150,157,164,170,181,187,188,195,202,209,216,223,224,231,238,250,
    5,11,18,34,43,52,62,73,84,94,104,114,115,125,134,143,144,151,158,165,176,182,183,189,196,203,210,217,218,225,232,251,
    0,6,12,26,35,44,53,63,74,85,95,105,106,116,126,135,136,145,152,159,171,177,178,184,190,197,204,211,212,219,226,245,
    1,2,7,19,27,36,45,54,64,75,86,96,97,107,117,127,128,137,146,153,166,172,173,179,185,191,198,205,206,213,220,239,
    3,13,20,28,37,46,55,65,76,87,88,98,108,118,119,129,138,160,167,227,233,
    8,14,21,29,38,47,56,66,77,78,89,99,109,110,120,234,240,
    9,15,22,30,39,48,57,67,68,79,90,100,241,246,
    23,31,40,49,58,69,80,247,252,
]

data_matrix_0 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)
data_matrix_1 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)
data_matrix_2 = np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan, dtype=np.float32)

heatmap_proc       = None
heatmap_proc_left  = None
heatmap_proc_right = None
heatmap_proc_out   = None

# ── PALETTE ───────────────────────────────────────────────────────────────────
bg        = "white"
teal_dark = "#0f6e56"
text_main = "#0a2a2a"

cmap = plt.get_cmap('viridis').copy()
cmap.set_bad(color='white')

if SHOW_BOTH_FEET:
    if UPSCALE_BLUR:

        # ── FIGURE ────────────────────────────────────────────────────────────
        # wide landscape: 6 panels on one row + colorbar on right
        fig = plt.figure(figsize=(22, 8), facecolor=bg)

        fig.text(0.5, 0.97,
                 "Real-time plantar pressure — crosstalk correction demo",
                 ha="center", va="top",
                 fontsize=26, fontweight="bold", color=teal_dark)

        # ── LAYOUT ────────────────────────────────────────────────────────────
        left_m    = 0.03
        right_m   = 0.08   # more right margin to keep colorbar inside
        cbar_w    = 0.015
        cbar_gap  = 0.015
        top       = 0.82   # lower top to give room for group labels
        bot       = 0.06
        col_gap   = 0.012
        group_gap = 0.045

        total_w = 1.0 - left_m - right_m - cbar_gap - cbar_w
        col_w   = (total_w - 4 * col_gap - group_gap) / 6.0
        h       = top - bot

        def x_pos(panel_idx):
            if panel_idx < 3:
                return left_m + panel_idx * (col_w + col_gap)
            else:
                return left_m + 3 * (col_w + col_gap) + group_gap + (panel_idx - 3) * (col_w + col_gap)

        ax_ll = fig.add_axes([x_pos(0), bot, col_w, h])
        ax_lr = fig.add_axes([x_pos(1), bot, col_w, h])
        ax_lo = fig.add_axes([x_pos(2), bot, col_w, h])
        ax_pl = fig.add_axes([x_pos(3), bot, col_w, h])
        ax_pr = fig.add_axes([x_pos(4), bot, col_w, h])
        ax_po = fig.add_axes([x_pos(5), bot, col_w, h])

        cbar_x = x_pos(5) + col_w + cbar_gap
        cax    = fig.add_axes([cbar_x, bot, cbar_w, h])

        # ── GROUP LABELS ──────────────────────────────────────────────────────
        raw_centre  = (x_pos(0) + x_pos(2) + col_w) / 2.0
        proc_centre = (x_pos(3) + x_pos(5) + col_w) / 2.0

        fig.text(raw_centre,  top, "Raw readout",
                 ha="center", va="bottom",
                 fontsize=17, fontweight="bold", color=teal_dark)
        fig.text(proc_centre, top, "Upscaled & smoothed",
                 ha="center", va="bottom",
                 fontsize=17, fontweight="bold", color=teal_dark)

        sep_x = (x_pos(2) + col_w + x_pos(3)) / 2.0
        fig.add_artist(plt.Line2D(
            [sep_x, sep_x], [bot, top + 0.07],
            transform=fig.transFigure,
            color=teal_dark, linewidth=0.8, linestyle='--', alpha=0.5))

        # ── SUBTITLES ─────────────────────────────────────────────────────────
        subtitle_kw = dict(fontsize=13, fontweight="bold", color=teal_dark, pad=8)
        ax_ll.set_title("Reference",          **subtitle_kw)
        ax_lr.set_title("Crosstalk-affected", **subtitle_kw)
        ax_lo.set_title("Model output",       **subtitle_kw)
        ax_pl.set_title("Reference",          **subtitle_kw)
        ax_pr.set_title("Crosstalk-affected", **subtitle_kw)
        ax_po.set_title("Model output",       **subtitle_kw)

        # ── HEATMAPS ──────────────────────────────────────────────────────────
        heatmap_0 = ax_ll.imshow(data_matrix_0, cmap=cmap,
                                 interpolation='nearest',
                                 vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_1 = ax_lr.imshow(data_matrix_1, cmap=cmap,
                                 interpolation='nearest',
                                 vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_2 = ax_lo.imshow(data_matrix_2, cmap=cmap,
                                 interpolation='nearest',
                                 vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_proc_left  = ax_pl.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_proc_right = ax_pr.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_proc_out   = ax_po.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)

        # ── BORDERS ───────────────────────────────────────────────────────────
        for ax in [ax_ll, ax_lr, ax_lo, ax_pl, ax_pr, ax_po]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # ── COLORBAR ──────────────────────────────────────────────────────────
        cb = fig.colorbar(heatmap_0, cax=cax)
        cb.set_label("Sensor resistance (Ω)",
                     fontsize=14, color=text_main, labelpad=12)
        cb.ax.yaxis.label.set_rotation(270)
        cb.ax.yaxis.label.set_va('bottom')
        cb.ax.tick_params(colors=text_main, labelsize=11)
        cb.set_ticks([0, 10000, 20000, 30000, 40000, 50000])
        cb.set_ticklabels(['0', '10k', '20k', '30k', '40k', '50k'])
        cb.outline.set_edgecolor(teal_dark)
        cb.outline.set_linewidth(1.0)

    else:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 5), facecolor=bg)
        heatmap_0 = ax0.imshow(data_matrix_0, cmap=cmap,
                               interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_1 = ax1.imshow(np.fliplr(data_matrix_1), cmap=cmap,
                               interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_2 = ax2.imshow(np.fliplr(data_matrix_2), cmap=cmap,
                               interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        ax0.set_title('Reference')
        ax1.set_title('Crosstalk-affected')
        ax2.set_title('Model output')
        plt.colorbar(heatmap_0, ax=ax0)
        plt.colorbar(heatmap_1, ax=ax1)
        plt.colorbar(heatmap_2, ax=ax2)

else:
    if UPSCALE_BLUR:
        fig, (ax0, ax_proc) = plt.subplots(1, 2, figsize=(10, 5), facecolor=bg)
        cmap_s = plt.get_cmap('viridis').copy()
        cmap_s.set_bad(color='white')
        if SINGLE_FOOT_SIDE.lower() == 'left':
            heatmap_0    = ax0.imshow(data_matrix_0, cmap=cmap_s,
                                      interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
            heatmap_proc = ax_proc.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap_s, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        else:
            heatmap_0    = ax0.imshow(np.fliplr(data_matrix_1), cmap=cmap_s,
                                      interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
            heatmap_proc = ax_proc.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap_s, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        plt.colorbar(heatmap_0,    ax=ax0)
        plt.colorbar(heatmap_proc, ax=ax_proc)
        ax0.set_title('Original')
        ax_proc.set_title('Upscaled & smoothed')
        heatmap_1 = None
    else:
        fig    = plt.figure(figsize=(8, 5), facecolor=bg)
        ax0    = fig.add_axes([0.15, 0.12, 0.7, 0.76])
        cmap_s = plt.get_cmap('viridis').copy()
        cmap_s.set_bad(color='white')
        if SINGLE_FOOT_SIDE.lower() == 'left':
            heatmap_0 = ax0.imshow(data_matrix_0, cmap=cmap_s,
                                   interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        else:
            heatmap_0 = ax0.imshow(np.fliplr(data_matrix_1), cmap=cmap_s,
                                   interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        plt.colorbar(heatmap_0, ax=ax0, fraction=0.046, pad=0.04)
        heatmap_1 = None


# ── DSP helpers ───────────────────────────────────────────────────────────────
def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma  = max(1e-6, float(sigma))
    radius = int(max(1, np.ceil(3 * sigma)))
    x      = np.arange(-radius, radius + 1)
    k      = np.exp(-(x**2) / (2 * sigma * sigma))
    return k / k.sum()

def _convolve1d_same(arr, kernel, axis):
    return np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode='same'), axis, arr)

def upscale_nn(data, factor):
    if factor <= 1:
        return data
    return np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)

def gaussian_blur_nanaware(data, sigma, factor):
    if np.all(np.isnan(data)):
        return data
    valid_mask     = ~np.isnan(data)
    adaptive_sigma = sigma * max(1.2, factor * 0.4)
    adaptive_sigma = max(0.8, min(adaptive_sigma, factor * 1.2))
    k              = _gaussian_kernel_1d(adaptive_sigma)
    data_filled    = np.where(valid_mask, data, 0.0)
    mask_float     = valid_mask.astype(float)
    tmp_data       = _convolve1d_same(data_filled, k, axis=1)
    blurred        = _convolve1d_same(tmp_data,    k, axis=0)
    tmp_mask       = _convolve1d_same(mask_float,  k, axis=1)
    mask_blurred   = _convolve1d_same(tmp_mask,    k, axis=0)
    eps            = 1e-8
    norm           = np.where(mask_blurred < eps, eps, mask_blurred)
    result         = blurred / norm
    result[~valid_mask] = np.nan
    return result

def pad_vertical(data, px=2):
    pad = np.full((px * UPSCALE_FACTOR, data.shape[1]), np.nan)
    return np.vstack([pad, data, pad])

def pad_raw(data, px=2):
    pad = np.full((px, data.shape[1]), np.nan)
    return np.vstack([pad, data, pad])


# ── Animation ─────────────────────────────────────────────────────────────────
def update_heatmap(frame):
    masked_0 = data_matrix_0.astype(float).copy()
    masked_1 = data_matrix_1.astype(float).copy()
    masked_2 = data_matrix_2.astype(float).copy()
    masked_0[image_pattern == 0] = np.nan
    masked_1[image_pattern == 0] = np.nan
    masked_2[image_pattern == 0] = np.nan

    right_display  = masked_1
    output_display = masked_2
    artists        = []

    if SHOW_BOTH_FEET:
        heatmap_0.set_data(pad_raw(masked_0))
        artists.append(heatmap_0)
        if heatmap_1 is not None:
            heatmap_1.set_data(pad_raw(right_display))
            heatmap_2.set_data(pad_raw(output_display))
            artists += [heatmap_1, heatmap_2]
        if UPSCALE_BLUR:
            if heatmap_proc_left is not None:
                proc_left = gaussian_blur_nanaware(
                    upscale_nn(masked_0, UPSCALE_FACTOR), BLUR_SIGMA, UPSCALE_FACTOR)
                heatmap_proc_left.set_data(pad_vertical(proc_left))
                artists.append(heatmap_proc_left)
            if heatmap_proc_right is not None:
                proc_right = gaussian_blur_nanaware(
                    upscale_nn(right_display, UPSCALE_FACTOR), BLUR_SIGMA, UPSCALE_FACTOR)
                heatmap_proc_right.set_data(pad_vertical(proc_right))
                artists.append(heatmap_proc_right)
            if heatmap_proc_out is not None:
                proc_out = gaussian_blur_nanaware(
                    upscale_nn(output_display, UPSCALE_FACTOR), BLUR_SIGMA, UPSCALE_FACTOR)
                heatmap_proc_out.set_data(pad_vertical(proc_out))
                artists.append(heatmap_proc_out)
        return tuple(artists)

    foot_data = masked_0 if SINGLE_FOOT_SIDE.lower() == 'left' else right_display
    heatmap_0.set_data(pad_raw(foot_data))
    if UPSCALE_BLUR and heatmap_proc is not None:
        proc = gaussian_blur_nanaware(
            upscale_nn(foot_data, UPSCALE_FACTOR), BLUR_SIGMA, UPSCALE_FACTOR)
        heatmap_proc.set_data(pad_vertical(proc))
        return heatmap_0, heatmap_proc
    return (heatmap_0,)


# ── Physics ───────────────────────────────────────────────────────────────────
def rsensor_from_adc(adc_counts):
    ADC_MAX  = (1 << 12) - 1
    vout     = adc_counts * 3.3 / ADC_MAX
    delta_v  = (vout - 0.825) / 1.0
    rs       = (1.6 - 0.46 - delta_v) / 33e-6 - 15_000.0
    rs       = np.where(vout == 0.0, 50_000.0, rs)
    return rs


# ── ML helpers ────────────────────────────────────────────────────────────────
def model_format(live_frame, mask):
    live_norm          = np.clip(live_frame, 0.0, RSENSOR_VMAX) / RSENSOR_VMAX
    frame_fin          = 1.0 - live_norm
    frame_fin[mask==0] = 0.0
    return frame_fin

def unet_live(live_frame, mask):
    x = np.array([model_format(live_frame, mask).astype(np.float32), mask])
    x = torch.from_numpy(x)[None, :, :, :].to(device)
    with torch.no_grad():
        y = model(x)
    return y.detach().cpu().numpy()

def vis_format(live_frame, mask):
    y         = unet_live(live_frame, mask)
    y_extract = y[0, 0, :, :]
    y_norm    = np.clip((1.0 - y_extract), 0.0, 1.0) * RSENSOR_VMAX
    y_norm[np.logical_not(mask)] = np.nan
    return y_norm


# ── UDP receiver ──────────────────────────────────────────────────────────────
def receive_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening on {UDP_IP}:{UDP_PORT}")

    def apply_lookup(src):
        out = np.zeros(len(src), dtype=int)
        for i, idx in enumerate(lookup_table):
            out[idx] = src[i]
        return out

    try:
        while True:
            data, _ = sock.recvfrom(BUFFER_SIZE)
            n_words       = 507 if not IMU_ENABLE else 513
            decimal_array = np.frombuffer(data, dtype=np.uint16).reshape(n_words)
            client_idx    = CLIENT_ID_SIZE if not IMU_ENABLE else CLIENT_ID_SIZE + IMU_DATA_SIZE
            client        = decimal_array[-client_idx]

            decimal_array1 = decimal_array[0:253]

            if client == 0:
                s1       = apply_lookup(decimal_array1)
                Rsensor  = rsensor_from_adc(s1)
                arr      = Rsensor.copy()
                arr[238] = 50000 if arr[238] >= 25000 else 0
                arr[251] = 50000 if arr[251] >= 15000 else 0
                frame_logger.log_frame(client=0, values=Rsensor)
                indices = np.argwhere(image_pattern == 1)
                for (i, j), v in zip(indices, arr):
                    data_matrix_0[i, j] = v if v > 0 else 0

            if client == 1:
                s1       = apply_lookup(decimal_array1)
                Rsensor  = rsensor_from_adc(s1)
                arr      = Rsensor.copy()
                arr[12]  = 50000 if arr[12]  >= 7000 else 0
                arr[179] = 50000 if arr[179] >= 7000 else 0
                frame_logger.log_frame(client=1, values=Rsensor)
                indices = np.argwhere(image_pattern == 1)
                for (i, j), v in zip(indices, arr):
                    data_matrix_1[i, j] = v if v > 200 else 0
                data_matrix_2[:, :] = vis_format(data_matrix_1, mask)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sock.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        data_thread        = Thread(target=receive_data)
        data_thread.daemon = True
        data_thread.start()

        ani = animation.FuncAnimation(
            fig, update_heatmap,
            interval=20, blit=True, cache_frame_data=False)
        plt.show()
    finally:
        frame_logger.close()
