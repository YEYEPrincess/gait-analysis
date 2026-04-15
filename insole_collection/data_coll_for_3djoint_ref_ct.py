"""
──────────────────────────────────────────────────────────────────────────────
Real-time Crosstalk-free Rsensor Heat-map
──────────────────────────────────────────────────────────────────────────────
* Listens on the configured UDP host and port for alternating data frames
  ├─ Frame-0 (client == 0) : IA gain 1×, V+ = 1.60 V  →  used to compute the
                           true sensor resistance matrix (Rsensor, Ω)
  └─ Frame-1 (client == 1) : raw high-gain data         →  displayed for reference

* Pipeline
  1.  Receive 507 × 16-bit ADC words per frame (last word = client id)
  2.  Re-order the 253 valid channels with **lookup_table**
  3.  Map the 1-D list into a 33 × 15 geometry using **image_pattern**
  4.  Frame-0 → `rsensor_from_adc()`                      # crosstalk-free R (Ω)
──────────────────────────────────────────────────────────────────────────────
"""
import os
from time import time
from pathlib import Path

from common.logger import FlatCsvFrameLogger
CSV_PATH = Path(__file__).resolve().parent / "common" / "insole_frames_flat.csv"
frame_logger = FlatCsvFrameLogger(str(CSV_PATH), n_channels=33*15)

import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import math
import torch

from models.unet import UNetSmall

MODEL_NAME = "unet100_5_sig.pt"

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/checkpoint" / MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TorchScript / archive PyTorch
model = UNetSmall().to(device)
state_dict = torch.load(str(MODEL_PATH), map_location=device)
model.load_state_dict(state_dict)
model.eval()

# UDP Configuration
UDP_IP = os.getenv("INSOLE_UDP_IP", "0.0.0.0")
UDP_PORT = int(os.getenv("INSOLE_UDP_PORT", "8999"))

# 2 bytes per sample
# Frame one size is 253
# Frame two size is 253
# 1 byte for client id
# 6 bytes for IMU data
Frame_ONE_SIZE = 253
Frame_TWO_SIZE = 253
IMU_ENABLE = False  # Enable IMU data reception
CLIENT_ID_SIZE = 1
IMU_DATA_SIZE = 6
BUFFER_SIZE = (Frame_ONE_SIZE + Frame_TWO_SIZE + CLIENT_ID_SIZE + IMU_DATA_SIZE) * 2  # Total buffer size in bytes

MATRIX_ROW = 33  # Matrix size
MATRIX_COLUMN = 15

image_pattern = [[0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
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
                ]
image_pattern = np.array(image_pattern)

mask = image_pattern.copy()
mask[0][7] = 0.0
mask[1][10] = 0.0
mask[3][4] = 0.0
mask[4][11] = 0.0

mask = (mask != 0.0).astype(np.float32)

# ---------- Display configuration ----------
# SHOW_BOTH_FEET = True  -> show both feet (two subplots)
# SHOW_BOTH_FEET = False -> show a single foot, chosen via SINGLE_FOOT_SIDE ("left" / "right")
SHOW_BOTH_FEET = True
SINGLE_FOOT_SIDE = 'right'  # 'left' or 'right'

# Common colorbar range (Ω)
RSENSOR_VMIN = 0
RSENSOR_VMAX = 50_000

# ---------- Single-foot postprocessed view (upscale + blur) configuration ----------
# When SHOW_BOTH_FEET=False and UPSCALE_BLUR=True:
#   left subplot = original, right subplot = postprocessed (upscaled + blurred)
UPSCALE_BLUR = True
UPSCALE_FACTOR = 10      # Upscaling factor (nearest neighbour)
BLUR_SIGMA = 1.5         # Base Gaussian sigma (will be adapted by factor)

lookup_table = [16,32,41,59,70,81,91,101,111,121,130,139,140,147,154,161,162,168,174,180,192,199,200,207,214,221,228,235,236,242,248,249,
                4,24,33,50,60,71,82,92,102,112,122,131,132,141,148,155,156,163,169,175,186,193,194,201,208,215,222,229,230,237,243,244,
                10,17,25,42,51,61,72,83,93,103,113,123,124,133,142,149,150,157,164,170,181,187,188,195,202,209,216,223,224,231,238,250,
                5,11,18,34,43,52,62,73,84,94,104,114,115,125,134,143,144,151,158,165,176,182,183,189,196,203,210,217,218,225,232,251,
                0,6,12,26,35,44,53,63,74,85,95,105,106,116,126,135,136,145,152,159,171,177,178,184,190,197,204,211,212,219,226,245,
                1,2,7,19,27,36,45,54,64,75,86,96,97,107,117,127,128,137,146,153,166,172,173,179,185,191,198,205,206,213,220,239,
                3,13,20,28,37,46,55,65,76,87,88,98,108,118,119,129,138,160,167,227,233,
                8,14,21,29,38,47,56,66,77,78,89,99,109,110,120,234,240,
                9,15,22,30,39,48,57,67,68,79,90,100,241,246,
                23,31,40,49,58,69,80,247,252
                ]

# Plotting Matrices
data_matrix_0 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)
data_matrix_1 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)
data_matrix_2 = np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan, dtype=np.float32)

# Handles for postprocessed heatmaps
heatmap_proc = None      # Single-foot postprocessed heatmap
heatmap_proc_left = None
heatmap_proc_right = None

if SHOW_BOTH_FEET:
    if UPSCALE_BLUR:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        cmap_left = plt.get_cmap('viridis').copy()
        cmap_left.set_bad(color='white')
        cmap_right = plt.get_cmap('viridis').copy()
        cmap_right.set_bad(color='white')
        cmap_out = plt.get_cmap('viridis').copy()
        cmap_out.set_bad(color='white')

        ax_ll, ax_lr, ax_lo = axes[0, 0], axes[0, 1], axes[0, 2]
        ax_pl, ax_pr, ax_po = axes[1, 0], axes[1, 1], axes[1, 2]

        heatmap_0 = ax_ll.imshow(data_matrix_0, cmap=cmap_left,
                                 interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_1 = ax_lr.imshow(data_matrix_1, cmap=cmap_right,
                                 interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_2 = ax_lo.imshow(data_matrix_2, cmap=cmap_out,
                                 interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_proc_left = ax_pl.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                         cmap=cmap_left, interpolation='nearest',
                                         vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_proc_right = ax_pr.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap_right, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_proc_out = ax_po.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap_out, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)

        ax_ll.set_title('Reference')
        ax_lr.set_title('Insole with crosstalk')
        ax_lo.set_title('Model output')
        ax_pl.set_title('Reference (Upscale + Blur)')
        ax_pr.set_title('Insole with crosstalk (Upscale + Blur)')
        ax_po.set_title('Model output (Upscale + Blur)')

        plt.colorbar(heatmap_0, ax=ax_ll)
        plt.colorbar(heatmap_1, ax=ax_lr)
        plt.colorbar(heatmap_2, ax=ax_lo)
        plt.colorbar(heatmap_proc_left, ax=ax_pl)
        plt.colorbar(heatmap_proc_right, ax=ax_pr)
        plt.colorbar(heatmap_proc_out, ax=ax_po)
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(2, 2, figsize=(10, 5))
        cmap0 = plt.get_cmap('viridis').copy()
        cmap0.set_bad(color='white')
        cmap1 = plt.get_cmap('viridis').copy()
        cmap1.set_bad(color='white')
        heatmap_0 = ax0.imshow(data_matrix_0, cmap=cmap0,
                               interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_1 = ax1.imshow(np.fliplr(data_matrix_1), cmap=cmap1,
                               interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        heatmap_2 = ax2.imshow(np.fliplr(data_matrix_2), cmap=cmap1,
                        interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        ax0.set_title('Left Foot')
        ax1.set_title('Right Foot (Mirrored)')
        plt.colorbar(heatmap_0, ax=ax0)
        plt.colorbar(heatmap_1, ax=ax1)
else:
    if UPSCALE_BLUR:
        # Single-foot mode + postprocessing comparison: left = original, right = postprocessed
        fig, (ax0, ax_proc) = plt.subplots(1, 2, figsize=(10, 5))
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad(color='white')
        if SINGLE_FOOT_SIDE.lower() == 'left':
            heatmap_0 = ax0.imshow(data_matrix_0, cmap=cmap,
                                   interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
            heatmap_proc = ax_proc.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        else:
            heatmap_0 = ax0.imshow(np.fliplr(data_matrix_1), cmap=cmap,
                                   interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
            heatmap_proc = ax_proc.imshow(np.full((MATRIX_ROW, MATRIX_COLUMN), np.nan),
                                          cmap=cmap, interpolation='nearest',
                                          vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        plt.colorbar(heatmap_0, ax=ax0)
        plt.colorbar(heatmap_proc, ax=ax_proc)
        ax0.set_title('Orig')
        ax_proc.set_title('Upscale + Blur')
        heatmap_1 = None
    else:
        # Single-foot mode without postprocessing: create a centered axis
        fig = plt.figure(figsize=(8, 5))
        ax0 = fig.add_axes([0.15, 0.12, 0.7, 0.76])
        cmap0 = plt.get_cmap('viridis').copy()
        cmap0.set_bad(color='white')
        if SINGLE_FOOT_SIDE.lower() == 'left':
            heatmap_0 = ax0.imshow(data_matrix_0, cmap=cmap0,
                                   interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        else:
            heatmap_0 = ax0.imshow(np.fliplr(data_matrix_1), cmap=cmap0,
                                   interpolation='nearest', vmin=RSENSOR_VMIN, vmax=RSENSOR_VMAX)
        plt.colorbar(heatmap_0, ax=ax0, fraction=0.046, pad=0.04)
        heatmap_1 = None


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """
    Generate a 1D Gaussian kernel with length 2*ceil(3*sigma) + 1.
    """
    sigma = max(1e-6, float(sigma))
    radius = int(max(1, np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel


def _convolve1d_same(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """
    1D convolution with 'same' output size along a given axis,
    implemented using np.apply_along_axis + np.convolve.
    """
    def _conv(v):
        return np.convolve(v, kernel, mode='same')
    return np.apply_along_axis(_conv, axis, arr)


def upscale_nn(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Nearest-neighbour upscaling; preserves NaN positions.
    """
    if factor <= 1:
        return data
    return np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)


def gaussian_blur_nanaware(data: np.ndarray, sigma: float, factor: int) -> np.ndarray:
    """
    Apply Gaussian blur to data that may contain NaNs.
    Smoothing is only performed over valid (non-NaN) regions, without
    spreading values into invalid regions.
    """
    if np.all(np.isnan(data)):
        return data
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return data

    # Adaptive sigma (heuristic based on upscaling factor)
    adaptive_sigma = sigma * max(1.2, factor * 0.4)
    adaptive_sigma = max(0.8, min(adaptive_sigma, factor * 1.2))

    k = _gaussian_kernel_1d(adaptive_sigma)
    data_filled = np.where(valid_mask, data, 0.0)
    mask_float = valid_mask.astype(float)

    # Separable convolution: rows first, then columns
    tmp_data = _convolve1d_same(data_filled, k, axis=1)
    blurred = _convolve1d_same(tmp_data, k, axis=0)

    tmp_mask = _convolve1d_same(mask_float, k, axis=1)
    mask_blurred = _convolve1d_same(tmp_mask, k, axis=0)

    eps = 1e-8
    norm = np.where(mask_blurred < eps, eps, mask_blurred)
    result = blurred / norm
    result[~valid_mask] = np.nan
    return result


def update_heatmap(frame):
    # Set non-sensor positions (mask=0) to NaN
    masked_0 = data_matrix_0.astype(float).copy()
    masked_1 = data_matrix_1.astype(float).copy()
    masked_2 = data_matrix_2.astype(float).copy()
    masked_0[image_pattern == 0] = np.nan
    masked_1[image_pattern == 0] = np.nan
    masked_2[image_pattern == 0] = np.nan

    right_display = masked_1
    output_display = masked_2
    artists = []

    if SHOW_BOTH_FEET:
        heatmap_0.set_data(masked_0)
        artists.append(heatmap_0)

        if heatmap_1 is not None:
            heatmap_1.set_data(right_display)
            heatmap_2.set_data(output_display)
            artists.append(heatmap_1)
            artists.append(heatmap_2)

        if UPSCALE_BLUR:
            if heatmap_proc_left is not None:
                proc_left = upscale_nn(masked_0, UPSCALE_FACTOR)
                proc_left = gaussian_blur_nanaware(proc_left, BLUR_SIGMA, UPSCALE_FACTOR)
                heatmap_proc_left.set_data(proc_left)
                artists.append(heatmap_proc_left)
            if heatmap_proc_right is not None:
                proc_right = upscale_nn(right_display, UPSCALE_FACTOR)
                proc_right = gaussian_blur_nanaware(proc_right, BLUR_SIGMA, UPSCALE_FACTOR)
                heatmap_proc_right.set_data(proc_right)
                artists.append(heatmap_proc_right)
            if heatmap_proc_out is not None:
                proc_out = upscale_nn(output_display, UPSCALE_FACTOR)
                proc_out = gaussian_blur_nanaware(proc_out, BLUR_SIGMA, UPSCALE_FACTOR)
                heatmap_proc_out.set_data(proc_out)
                artists.append(heatmap_proc_out)

        return tuple(artists)

    # Single-foot mode: original + (optional) postprocessed view
    if SINGLE_FOOT_SIDE.lower() == 'left':
        foot_data = masked_0
    else:
        foot_data = right_display

    heatmap_0.set_data(foot_data)

    if UPSCALE_BLUR and heatmap_proc is not None:
        proc = upscale_nn(foot_data, UPSCALE_FACTOR)
        proc = gaussian_blur_nanaware(proc, BLUR_SIGMA, UPSCALE_FACTOR)
        heatmap_proc.set_data(proc)
        return heatmap_0, heatmap_proc

    return (heatmap_0,)


def rsensor_from_adc(adc_counts: np.ndarray) -> np.ndarray:
    """
    Convert 12-bit ADC values (sorted_one_d_array1) into crosstalk-free
    sensor resistance Rsensor (Ω).

    —— IA Current Source (Frame-1) ——————————
      ADC Reference       VADC_REF   = 3.3   V
      IA Reference        VREF_IA    = 0.825 V
      IA Gain             GAIN       = 1.0
      IA Positive input   VDAC2      = 1.6   V
      Diode Forward drop  V_DIODE    = 0.46  V
      Current Source      I_OUT      = 33 µA
      Series Bias R       RS_OFFSET  = 15 kΩ
      ADC=0 Fill value    RS_INVALID = 50 kΩ
    """

    # 0) Output voltage Vout --------------------------------------------------
    ADC_MAX    = (1 << 12) - 1                      # 4095
    VADC_REF   = 3.3
    vout = adc_counts * VADC_REF / ADC_MAX

    # 1) ΔV across instrumentation amplifier --------------------------------
    VREF_IA = 0.825
    GAIN    = 1.0
    delta_v = (vout - VREF_IA) / GAIN

    # 2) Optional negative saturation limit (disabled for now)
    # SAT_NEG = -0.8
    # delta_v_mag = np.abs(np.clip(delta_v, SAT_NEG, None))

    # 3) Rsensor computation --------------------------------------------------
    VDAC2     = 1.6
    V_DIODE   = 0.46
    I_OUT     = 33e-6
    RS_OFFSET = 15_000.0

    rs = (VDAC2 - V_DIODE - delta_v) / I_OUT - RS_OFFSET
    # rs = (VDAC2 - V_DIODE - delta_v_mag) / I_OUT - RS_OFFSET

    # 4) Special case: ADC reads 0 → 50 kΩ, and clip any negative values -----
    RS_INVALID = 50_000.0
    rs = np.where(vout == 0.0, RS_INVALID, rs)

    return rs

def receive_data():
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

    try:
        while True:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            # Convert raw data into a 1D array of 16-bit integers
            if not IMU_ENABLE:
                decimal_array = np.frombuffer(data, dtype=np.uint16).reshape(507)   # 507 words without IMU
            else:
                decimal_array = np.frombuffer(data, dtype=np.uint16).reshape(513)   # 513 words with IMU

            if not IMU_ENABLE:
                client = decimal_array[-(CLIENT_ID_SIZE)]  # Extract client ID (0 or 1), index 506
            else:
                client = decimal_array[-(CLIENT_ID_SIZE + IMU_DATA_SIZE)]  # Index 506 with IMU enabled

            decimal_array1 = decimal_array[0:253]
            decimal_array2 = decimal_array[253:506]

            if IMU_ENABLE:
                imu_data_array = decimal_array[507:513]

            # Left image (client == 0)
            if client == 0:
                sorted_one_d_array1 = np.zeros(len(decimal_array1), dtype=int)  # mapped 1D list
                sorted_one_d_array2 = np.zeros(len(decimal_array2), dtype=int)  # mapped 1D list

                for i, index in enumerate(lookup_table):
                    # Frame-1: IA gain 1.0, VDAC2 = 1.6 V
                    sorted_one_d_array1[index] = decimal_array1[i]

                for i, index in enumerate(lookup_table):
                    # Frame-1: high-gain path (not used in Rsensor conversion here)
                    sorted_one_d_array2[index] = decimal_array2[i]

                # Crosstalk-free Rsensor (Ω) for Frame-1
                Rsensor_array = rsensor_from_adc(sorted_one_d_array1)

                # Use Rsensor as the mapped 1D array
                sorted_one_d_array = Rsensor_array

                # Correcting offset with binary
                max = 50000
                min = 0
                thr = 25000
                sorted_one_d_array[238] = max if sorted_one_d_array[238] >= thr else min

                # Map sorted 1D list to 33×15 matrix
                indices = np.argwhere(image_pattern == 1)
                for (i, j), value in zip(indices, sorted_one_d_array):
                    if value > 0:  # Minimum accepted value
                        data_matrix_0[i, j] = value
                    else:
                        data_matrix_0[i, j] = 0

                                # Logging data
                frame_logger.log_frame(client=0, values=data_matrix_0.flatten().tolist())

                

            # Right image (client == 1)
            if client == 1:
                sorted_one_d_array1 = np.zeros(len(decimal_array1), dtype=int)  # mapped 1D list
                sorted_one_d_array2 = np.zeros(len(decimal_array2), dtype=int)  # mapped 1D list

                for i, index in enumerate(lookup_table):
                    sorted_one_d_array1[index] = decimal_array1[i]

                for i, index in enumerate(lookup_table):
                    sorted_one_d_array2[index] = decimal_array2[i]

                Rsensor_array = rsensor_from_adc(sorted_one_d_array1)  # Crosstalk-free Rsensor (Ω)

                sorted_one_d_array = Rsensor_array

                # Correcting offset with binary
                max = 50000
                min = 0
                thr = 7000
                sorted_one_d_array[12] = max if sorted_one_d_array[12] >= thr else min
                sorted_one_d_array[179] = max if sorted_one_d_array[179] >= thr else min

                # Map sorted 1D list to 33×15 matrix
                indices = np.argwhere(image_pattern == 1)
                offset = 30000  # currently unused
                for (i, j), value in zip(indices, sorted_one_d_array):
                    if value > 200:  # Minimum accepted value
                        data_matrix_1[i, j] = value
                    else:
                        data_matrix_1[i, j] = 0

                data_matrix_2temp = unet_live(data_matrix_1, mask)

                data_matrix_2[:,:]=vis_format(data_matrix_2temp, mask)

                save_csv = True
                if save_csv:
                    csv_values = csv_format(data_matrix_2temp)
                    frame_logger.log_frame(client=1, values=csv_values)


    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sock.close()

def model_format(live_frame, mask):

    # ---------- Normalize to [0, 1] ----------
    MAX_R = 50000.0
    live_norm = np.clip(live_frame, 0.0, MAX_R) / MAX_R

    # invert sensor values: 1 = pressed, 0 = not pressed
    frame_fin = 1.0 - live_norm

    frame_fin[mask==False] = 0.0

    return frame_fin

def unet_live(live_frame, mask):
    formatted_frame=model_format(live_frame,mask).astype(np.float32)

    x = [formatted_frame,mask]

    x=np.array(x)

    x = torch.from_numpy(x)[None, :, :, :]
    x = x.to(device)

    with torch.no_grad():
        y = model(x)
    
    y = y.detach().cpu().numpy()

    y_extract = y[0, 0, :, :]

    y = (1.0 - y_extract)*50000

    return y

def vis_format(y, mask):
    y_vis = y.copy()
    inactive_mask = np.logical_not(mask)
    y_vis[inactive_mask] = np.nan
    return y_vis

def csv_format(matrix):
    return matrix.flatten()

if __name__ == "__main__":
    try:
        data_thread = Thread(target=receive_data)
        data_thread.daemon = True
        data_thread.start()

        ani = animation.FuncAnimation(fig, update_heatmap,
                                    interval=20, blit=True, cache_frame_data=False)
        plt.show()
    finally:
        # Make sure CSV file is properly closed
        frame_logger.close()
