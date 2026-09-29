# Lower Limb Kinematics Estimation from Smart Insoles

This repository contains the research code for an MSc dissertation investigating whether wearable smart-insole measurements can reconstruct continuous lower-limb motion without camera input during model inference. Bilateral plantar-pressure maps and foot-mounted inertial measurement units (IMUs) are used to estimate the 3D positions of eight landmarks. Monocular video processed with SAM 3D Body provides supervision during development.

The dissertation compares a pressure-only CNN-LSTM, pressure-IMU concatenation, and feature-wise linear modulation (FiLM). This release provides one model-training entry point, [FiLM_cnn_lstm_imu.ipynb](FiLM_cnn_lstm_imu.ipynb), using three-participant leave-one-subject-out (LOSO) cross-validation. The historical `experiment/` notebooks are excluded from this upload. This is a research prototype with video-derived reference poses, rather than a validated replacement for optical motion capture.

The released notebook uses seed 42 and includes MPJPE, constant-pose, motion-fidelity, per-fold IMU-zeroing and FiLM-modulation evaluation. Its execution counts and saved outputs are cleared. The results section presents FiLM seed-42 position errors and the IMU-zeroing MAE results.

## Research question

Plantar pressure captures foot-ground loading but provides limited direct information about the motion of the knee and hip, particularly when the foot is unloaded during swing. The project investigates whether temporal learning and inertial conditioning can recover lower-limb trajectories from compact foot-worn sensing, and whether the learned mapping transfers to an unseen participant.

The eight output landmarks, in order, are:

1. Left hip
2. Right hip
3. Left knee
4. Right knee
5. Left heel
6. Right heel
7. Left big-toe tip
8. Right big-toe tip

Targets are expressed in pelvis-relative, orientation-normalised coordinates. The task estimates relative lower-limb pose, not global walking translation. Angles derived from the landmarks are geometric proxies; the knee-heel-toe angle is not an anatomical ankle angle.

## Method

```text
Bilateral pressure                  Bilateral six-axis IMU
        |                                      |
Pressure preprocessing              Pressure-derived stance detection
        |                           19 features per foot, including ZUPT
Frame-wise 2D CNN                              |
256-dimensional embedding           64-dimensional embedding
        |                                      |
        +--------- FiLM or concatenation -------+
                             |
                   Two-layer LSTM, width 256
                             |
                   Scalar temporal attention
                             |
                   3D landmark regression

RGB video -> SAM 3D Body -> levelling / normalisation / synchronisation
                                      |
                              Supervision and evaluation
```

### Pressure and video preprocessing

- Each insole has a nominal 253-channel layout embedded in a `33 x 15` grid.
- Pressure maps are bilinearly upsampled to `66 x 30` and smoothed with mask-normalised Gaussian filtering. Non-finite values are replaced before model input. Upsampling adds no physical sensor measurements.
- Smartphone video is decimated from approximately 30 fps to approximately 15 fps. SAM 3D Body supplies 3D landmarks; session-level levelling estimates a common rotation from pressure-selected contact points and hip geometry.
- Synchronisation uses joint timestamps as the reference and nearest wearable samples within 150 ms. For the three multimodal sessions, rows require both pressure and IMU matches.

### Multimodal temporal model

The pressure encoder shares CNN weights across frames and produces a 256-dimensional embedding. Each foot's raw six-axis IMU is expanded inside the multimodal notebooks to 19 features, including acceleration, scaled angular velocity, normalised acceleration direction, ZUPT-corrected velocity, within-step displacement and sine/cosine gait phase. Both feet are embedded into 64 dimensions.

FiLM generates a per-frame scale and bias from the IMU embedding:

```text
fused = (1 + gamma) * pressure_embedding + beta
```

The concatenation control instead joins the pressure and IMU embeddings, giving a 320-dimensional LSTM input. Both multimodal models use a two-layer unidirectional LSTM with hidden width 256, dropout 0.3, and learned scalar temporal-attention pooling.

The input window contains 30 frames. The multimodal regression head outputs the final **three poses within that window**, with only the last pose scored. The earlier two poses support trajectory losses. This is retrospective reconstruction, not future forecasting.

The multimodal training objective is:

```text
L = L_kinematic + 0.5 L_contact + 0.3 L_bone + 0.1 L_smooth
```

The kinematic term is weighted L1, with landmark weights 0.5 for hips, 1 for knees and 2 for heels/toes, and frame weights `(0.5, 0.5, 1.0)`. Contact and smoothness penalties operate on normalised predictions; the bone term uses inverse-normalised coordinates and development-set median lengths.

## Repository guide

| Entry | Purpose |
|---|---|
| [FiLM CNN-LSTM notebook](FiLM_cnn_lstm_imu.ipynb) | Single model-training entry point; seed 42, cleared outputs, integrated position/motion metrics and per-fold IMU ablation |
| [Preprocessing notebook](part2_preprocessing.ipynb) | Pressure resampling, landmark selection, normalisation and synchronisation |
| [Crosstalk notebook](part1-CT.ipynb) | Legacy U-Net crosstalk model development |
| `insole_collection/` | Sensor collection, conversion, levelling and synchronisation tools |
| `joint_collection/joint_processing/` | Video frame calibration, SAM inference and joint-session merging |
| `joint_collection/sam-3d-body/` | Third-party SAM 3D Body source and its own documentation/license |

Use `FiLM_cnn_lstm_imu.ipynb` as the model-training entry point for this release. The local `experiment/` directory is excluded from the upload.

## Environment setup

Clone the repository:

```bash
git clone https://github.com/YEYEPrincess/gait-analysis.git
cd gait-analysis
```

Two Conda environment exports are provided:

```bash
conda env create -f insole_collection/insole_environment.yml
conda env create -f joint_collection/joint_environment.yml
```

- `ML`: insole processing and model development.
- `sam_3d_body`: video processing and SAM 3D Body inference.

Both exports contain Python 3.11 and platform-specific Windows dependencies. They record the development environments; installation on other platforms requires adaptation. A clean-environment installation was not tested during this documentation update. FFmpeg is used for video tools, and a CUDA-capable GPU is useful for SAM inference and training.

For notebook use in the model environment, install/register a kernel if it is not already available:

```bash
conda activate ML
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name gait-analysis-ml --display-name "Gait analysis ML"
python -m jupyterlab
```

The model notebooks require NumPy, SciPy, PyTorch and Matplotlib; some visualisation cells additionally require ImageIO and tqdm. Check these imports in the selected kernel before execution. See the third-party [SAM installation guide](joint_collection/sam-3d-body/INSTALL.md) for its additional dependencies and model access instructions.

The video-supervision tools require the following external weights:

```text
joint_collection/checkpoints/sam-3d-body-dinov3/model.ckpt
joint_collection/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

These weights are not supplied by this source-code update. The pretrained gait checkpoints also require a verified association with the model configuration, fold, data version and normalisation statistics before reuse.

## Data

Raw participant recordings and processed training datasets are not included in the proposed source-code release. No public dataset download is provided here; execution requires the corresponding study data supplied separately by the project maintainer. Do not treat third-party example images as the gait study dataset.

The study uses **5,756 synchronised frames from three participants**, evaluated with participant-level LOSO cross-validation.

Each `<session>_synced.npz` contains at least:

| Key | Shape | Meaning |
|---|---|---|
| `insoles` | `(T, 2, 66, 30)` | Left/right pressure representations |
| `imu` | `(T, 2, 6)` | Acceleration in g and angular velocity in degrees/s |
| `joints` | `(T, 8, 3)` | Reference landmark coordinates in metres |
| `timestamps` | `(T,)` | Shared reference timestamps in milliseconds |
| `joint_names` | `(8,)` | Landmark names; stored as an object array in these files |

The 19-dimensional IMU features are computed from the raw six-axis measurements in the multimodal training notebook.

### Starting from synchronised data

1. Place the three synchronised study sessions under `data/synced_levelled/`, the input directory configured in the notebook.
2. Open [FiLM_cnn_lstm_imu.ipynb](FiLM_cnn_lstm_imu.ipynb) at the repository root and select the `Gait analysis ML` kernel.
3. Ensure the **kernel working directory is the repository root** before running the setup cells. Alternatively, set `BASE` in your working copy to an absolute data path.
4. Execute the setup, loading, preprocessing, model definitions, training helpers and LOSO loop in order. The notebook retains `SEED=42`, `USE_FILM=True` and `PRED_FRAMES=3` from the original seed-42 notebook.
5. Run the aggregate MAE/MSE cell, then **10.1 Extended evaluation: position and motion diagnostics**. It computes overall, distal and per-landmark MPJPE, the development-participant constant-pose baseline, amplitude ratios and Pearson correlations.
6. Run **10.2 Complete per-fold IMU ablation and FiLM diagnostics** with the matching checkpoints in `MODELS_DIR`. Each fold is evaluated with full and zero IMU input, using the same weights and target-normalisation statistics. Missing checkpoints are reported and skipped; all three are needed for a complete summary. No extra evaluation script is required.
7. Loss curves and trajectory visualisations are optional. All execution counts and outputs remain cleared; no new study results are bundled.

Before training, set `MODELS_DIR` to the directory where checkpoints should be saved. Choose a separate directory for each run to preserve existing checkpoints.

### Starting from raw recordings

This path requires session-specific calibration and local path configuration. The selected update starts from prepared paired pressure arrays for pressure processing; the CSV schema adapter does not construct those arrays. An older converter already tracked in the repository is not updated or validated by this release.

| Stage | Entry points | Output or prerequisite |
|---|---|---|
| Pressure input preparation | `adapt_csv_schema.py` (CSV schema only) | Prepare paired pressure NPZ separately; the local raw-pressure converter is excluded from this update |
| IMU conversion | `offline_processing_imu.py` | Use its `--in` / `--out` arguments for explicit CSV and output paths |
| Video preparation | `frame_calibration.py`, `extract_frames.py` | Select the start frame and keep the frame rate consistent |
| Video supervision | `joint_extract.py`, `merge_npz.py` | SAM joint arrays; configure `VIDEO_NAMES` and required weights |
| Session staging | `finalize_session.py` | Stage pressure/IMU under `data/`; choose any trimming consistently across modalities |
| Base preprocessing | `part2_preprocessing.ipynb` | `data/insoles_preprocessed/`, selected joints and `data/synced/` |
| Camera levelling | `level_ground_joints.py` | Levelled joint arrays; synchronisation is a separate preprocessing step |

The repository contains legacy UDP capture scripts. The paired pressure/IMU CSV workflow also references an external BALANCE collector; that collector/firmware is not included in this release. Hardware capture therefore requires the matching acquisition setup in addition to this repository.

## Experiment design and evaluation

The following configuration describes the released FiLM notebook with seed 42.

- Three outer LOSO folds: one held-out participant per fold, with model weights reinitialised.
- Development windows have length 30 and stride 1; a shuffled 15% is used for validation and checkpoint selection.
- Adjacent windows overlap, so the internal validation set is not an independent participant-level generalisation test.
- Normalisation statistics are computed from the two non-test participants, including their internal validation frames; the outer test participant is excluded.
- Seed 42, batch size 32, Adam, weight decay `1e-5`, maximum 700 epochs and early-stopping patience 400.
- CNN learning rate `2e-3`, LSTM `3e-3`, other parameter groups `1e-3`. The multimodal training helper clips gradient norm at 1.0.
- MAE and MSE are calculated after inverse normalisation, in metres and square metres. MPJPE averages the 3D Euclidean landmark error, reported in millimetres.
- MAE/MSE and per-landmark summaries use unweighted fold means and population standard deviations (`ddof=0`); these are not repeated-seed significance estimates.

The integrated MPJPE cell runs after the LOSO loop and uses inverse-normalised `fold_results[...]["preds"]` and `fold_results[...]["trues"]`. It reports millimetres, gives each fold equal weight in the overall summary, and excludes both hips for distal MPJPE. Within-fold distribution statistics describe the window–landmark Euclidean errors; the per-landmark table summarises means and population standard deviations across folds.

The extended evaluation also compares against the non-test participants' mean pose and reports per-coordinate amplitude ratios and Pearson correlations. The checkpoint-based IMU-zeroing evaluation reports changes in overall and distal MPJPE without retraining. FiLM statistics match the seed-123 implementation: the hook collects both full-input and zero-IMU passes, and the reported values are unweighted averages of batch summaries, rather than full-input-only statistics. This loader reconstruction uses the default `NORMALISE_IMU=False` configuration.

## FiLM results (seed 42)

The following full-input results summarise FiLM with seed 42 under three-participant LOSO evaluation. MAE is reported in metres and MPJPE in millimetres after inverse normalisation, with equal weighting across the three held-out participants.

| Model | Seed | MAE (m) | MPJPE (mm) |
|---|---:|---:|---:|
| FiLM CNN-LSTM | 42 | 0.0532 | 117.9 |

### Test-time IMU-zeroing ablation

Table 5 of the dissertation reports the following MAE ablation results for the seed-42 FiLM model.

| Model | Seed | Full-input MAE (m) | Zero-IMU MAE (m) | Increase |
|---|---:|---:|---:|---:|
| FiLM CNN-LSTM | 42 | 0.0532 | 0.0612 | 15.0% |

The IMU-zeroing experiment replaces the test-time IMU tensor with zeros while keeping the trained model weights fixed. The increase in MAE indicates sensitivity to removal of the inertial input. No retraining is performed; the change can reflect both loss of inertial information and the shift to an all-zero input distribution.

## Limitations

The study uses three participants and video-estimated reference poses. Adjacent windows overlap, so window-level errors are temporally correlated. The fold statistics describe performance across the three held-out participants; larger cohorts and repeated training runs are needed to assess broader generalisation and robustness.

The role of FiLM in individual gait phases remains a subject for further analysis. Phase-conditioned errors and inspection of the learned scale and bias parameters would help explain when inertial conditioning contributes to reconstruction accuracy. The current study does not establish clinical validity or end-to-end real-time deployment.

## Acknowledgements

This repository is a fork of [BioZ-HIVE/gait-analysis](https://github.com/BioZ-HIVE/gait-analysis), originating from [omargdr/gait-analysis](https://github.com/omargdr/gait-analysis). Video supervision uses SAM 3D Body; its source, attribution and license are retained under [joint_collection/sam-3d-body](joint_collection/sam-3d-body/README.md). FiLM follows feature-wise affine conditioning, and ZUPT follows zero-velocity-update principles. Third-party software and weights retain their respective terms.
