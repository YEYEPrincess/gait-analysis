# Gait Analysis with Smart Insole

## Overview

This project develops an end-to-end pipeline for gait analysis using a low-cost smart insole and a monocular camera. The goal is to enable accessible, in-the-wild biomechanical analysis without relying on laboratory motion capture systems.

The system combines plantar pressure sensing with video-based 3D joint estimation and machine learning models to reconstruct and predict lower-limb motion.

---

## Motivation

Traditional gait analysis systems are:
- expensive
- not portable
- limited to controlled environments

This project explores a more scalable alternative using:
- wearable pressure sensors
- standard smartphone video

Target applications include:
- rehabilitation
- sports performance analysis
- mobility monitoring

---

## Pipeline

The pipeline is composed of the following stages:

### 1. Data Acquisition
- Pressure data from a high-resolution insole (253 sensors)
- Video recorded with a smartphone
- Parallel recording of both modalities

### 2. Preprocessing
- Conversion of raw sensor data into 33×15 pressure maps
- Handling of missing or faulty sensors
- Temporal alignment between pressure data and video frames

### 3. Crosstalk Correction
- A U-Net model is used to correct spatial distortions in pressure measurements
- Produces cleaner and more structured pressure maps

### 4. 3D Joint Extraction
- SAM 3D Body is used to extract 3D joint trajectories from monocular video
- Provides supervision without requiring RGB-D systems

### 5. Joint Prediction
- A CNN-LSTM model predicts lower-limb joint coordinates from pressure data
- Temporal modelling captures gait dynamics over sequences

---

## Model Architecture

- **U-Net**: spatial correction of pressure maps  
- **CNN encoder**: feature extraction per frame  
- **LSTM**: temporal modelling  
- **Output**: 3D coordinates of lower-limb joints  

---

## Installation

This repository uses two separate Python environments:

- `insole_collection/insole_environment.yml` for smart-insole capture, preprocessing, crosstalk correction, and visualisation.
- `joint_collection/joint_environment.yml` for video frame processing and SAM 3D Body joint extraction.

### Prerequisites

Install the following before creating the environments:

- Git
- Anaconda or Miniconda
- FFmpeg, available on your system `PATH`
- A CUDA-capable GPU is recommended for SAM 3D Body inference. CPU execution may be very slow.

Check FFmpeg is available:

```bash
ffmpeg -version
ffprobe -version
```

### Clone the Repository

```bash
git clone <repository-url>
cd gait-analysis
```

### Insole Environment

Create and activate the environment used by the insole scripts:

```bash
conda env create -f insole_collection/insole_environment.yml
conda activate ML
```

Quick import check:

```bash
python -c "import numpy, pandas, cv2, torch; print('insole environment ready')"
```

The live insole scripts bind to `0.0.0.0:8999` by default. To use a specific network adapter or port, set:

PowerShell:

```powershell
$env:INSOLE_UDP_IP = "0.0.0.0"
$env:INSOLE_UDP_PORT = "8999"
```

Windows Command Prompt:

```bash
set INSOLE_UDP_IP=0.0.0.0
set INSOLE_UDP_PORT=8999
```

On macOS/Linux:

```bash
export INSOLE_UDP_IP=0.0.0.0
export INSOLE_UDP_PORT=8999
```

Generated insole data is intentionally ignored by Git. Runtime CSV captures are written to:

```text
insole_collection/common/insole_frames_flat.csv
```

Processed datasets are written under:

```text
insole_collection/data_ct/
insole_collection/data_for_3djoint/
```

### Joint Extraction Environment

Create and activate the environment used by the video and SAM 3D Body scripts:

```bash
conda env create -f joint_collection/joint_environment.yml
conda activate sam_3d_body
```

Quick import check:

```bash
python -c "import cv2, numpy, torch; print('joint environment ready')"
```

### SAM 3D Body Checkpoints

Model weights are not committed to the repository. Place the SAM 3D Body checkpoint files in:

```text
joint_collection/checkpoints/sam-3d-body-dinov3/model.ckpt
joint_collection/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

The joint extraction code expects this layout when running:

```bash
python joint_collection/joint_processing/joint_extract.py
```

### Expected Data Locations

Input videos for joint extraction should be placed in:

```text
joint_collection/joint_processing/input_videos/
```

Extracted frames are generated in:

```text
joint_collection/joint_processing/output_frames/
```

Joint outputs, overlays, and merged sessions are written under:

```text
joint_collection/joint_processing/outputs/
```

These generated folders are ignored by Git to keep the repository lightweight.

## Operating Guide

The insole and joint-extraction parts use different Conda environments and should be run separately.

- During the experiment, run only the insole collection script in the `ML` environment and record the video at the same time with the camera.
- After the experiment, copy the recorded video into `joint_collection/joint_processing/input_videos/`.
- Then switch to the `sam_3d_body` environment for frame extraction and 3D joint extraction.

### 1. During the Experiment: Insole Capture

Activate the insole environment:

```bash
conda activate ML
```

Choose one collection script depending on the data you need.

#### Crosstalk Suppression Training Data

Use this when collecting data to train or evaluate the crosstalk suppression model:

```bash
python insole_collection/data_coll_ct.py
```

This mode is for the setup where two insoles are superimposed:

- one stream is treated as the reference signal
- the other stream is treated as the crosstalk-affected signal
- the resulting raw CSV is later converted into paired `X` and `Y` arrays for crosstalk model training

After collection, convert the captured CSV into an `.npz` dataset:

```bash
python insole_collection/offline_processing_ct.py
```

Preview the processed crosstalk dataset with:

```bash
python insole_collection/visualise_ct.py
```

#### Joint Prediction Training Data

Use one of these when collecting insole data that will later be aligned with 3D joint trajectories.

```bash
python insole_collection/data_coll_for_3djoint.py
```

This version is intended for live collection of processed insole maps for joint prediction. It displays raw left/right insole maps and no-crosstalk outputs. In the current configuration, `client == 0` is passed through the U-Net, while `client == 1` uses the bypass path unless `BOTH_CROSSTALK` is enabled in the script.

```bash
python insole_collection/data_coll_for_3djoint_ref_ct.py
```

This version is intended for a reference/crosstalk setup during joint-data collection:

- `client == 0` is logged as the reference 33x15 map
- `client == 1` is passed through the U-Net
- the display compares reference, crosstalk-affected input, and model output

After collection, convert the captured CSV into the joint-prediction insole dataset:

```bash
python insole_collection/offline_processing_for_3djoint.py
```

Preview the result with:

```bash
python insole_collection/visualise_for_3djoint.py
```

For synchronisation with the video-derived joint data, enable trimming in `visualise_for_3djoint.py`:

```python
ENABLE_TRIM_AFTER_PREVIEW = True
```

This lets you choose the first insole frame to keep after previewing the pressure maps. The saved cropped `.npz` has timestamps rebased from zero, which makes alignment with the calibrated video frames easier.

If you have several crosstalk-training sessions, merge them with:

```bash
python insole_collection/merge_npz.py
```

Note: `insole_collection/merge_npz.py` expects datasets with both `X` and `Y` arrays, so it is suited to the crosstalk-training datasets produced by `offline_processing_ct.py`.

#### Live Demonstration

To demonstrate live crosstalk correction without running the full collection workflow:

```bash
python insole_collection/demo_ct.py
```

### 2. After the Experiment: Video Placement

Copy the video recorded during the experiment into:

```text
joint_collection/joint_processing/input_videos/
```

Keep the filename matched to the `video_path` value used in `frame_calibration.py`. For example:

```python
video_path = "sample_video_1.mov"
```

The folder is present in the repository via `.gitkeep`, but video files inside it are ignored by Git.

### 3. Joint Frame Extraction and Synchronisation

Activate the joint environment:

```bash
conda activate sam_3d_body
```

Extract frames and choose the first frame for synchronisation:

```bash
python joint_collection/joint_processing/frame_calibration.py
```

This script:

- reads the video from `joint_collection/joint_processing/input_videos/`
- extracts frames at the configured FPS
- asks for the first frame to keep
- renumbers the kept frames from timestamp zero

The extracted/calibrated frames are written to:

```text
joint_collection/joint_processing/output_frames/
```

### 4. 3D Joint Extraction

Run SAM 3D Body on the calibrated frames:

```bash
python joint_collection/joint_processing/joint_extract.py
```

This writes per-frame joint `.npz` files under:

```text
joint_collection/joint_processing/outputs/joints_npz/
```

Finally, merge each session into a single joint `.npz`:

```bash
python joint_collection/joint_processing/merge_npz.py
```

Merged joint sessions are written under:

```text
joint_collection/joint_processing/outputs/sessions_final/
```

## Research Notebooks

The root-level notebooks contain the bulk of the modelling and experimentation work:

- `part1-CT.ipynb`: crosstalk suppression model development and training workflow.
- `part2_preprocessing.ipynb`: preprocessing steps for preparing paired insole and 3D-joint datasets.
- `part2-cnn_lstm_loso.ipynb`: CNN-LSTM joint prediction experiments, including leave-one-subject-out style evaluation.

These notebooks were originally developed in Google Colab. They are included as research records and working references, but they are not expected to run directly in this local repository without edits.

Before running them locally, expect to update:

- dataset paths
- checkpoint paths
- Google Drive or Colab-specific file access
- runtime/device assumptions
- any notebook-only installation cells

The scripts in `insole_collection/` and `joint_collection/` are the cleaner local operating pipeline. The notebooks are best treated as the experimental training layer built on top of the generated `.npz` datasets.

### Notes

- If `conda env create` reports that an environment already exists, update it with:

```bash
conda env update -f insole_collection/insole_environment.yml --prune
conda env update -f joint_collection/joint_environment.yml --prune
```

- If PyTorch or Detectron2 installation fails, reinstall the versions that match your CUDA, Python, and operating system setup.
- Large files such as videos, extracted frames, `.npz` datasets, model checkpoints, and local Conda environments should remain untracked.
