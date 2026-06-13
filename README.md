# Deep Learning Model for Prediction of Rat Sleep Staging Based on EEG Signals

Automatic sleep stage classification (Wake / NREM / REM) from rat EEG and EMG recordings using four deep learning architectures. Built on the openly available **OpenNeuro ds006366** dataset (92 subjects, 148 recordings).

---

## Results

| Model | Accuracy | Balanced Acc | Macro F1 | Cohen's κ | AUC-ROC | REM F1 |
|---|---|---|---|---|---|---|
| Baseline MLP | 0.746 | 0.749 | 0.663 | 0.569 | 0.896 | 0.435 |
| 1D CNN | 0.841 | 0.819 | 0.762 | 0.720 | 0.941 | 0.545 |
| **CNN + LSTM** | 0.862 | 0.858 | **0.808** | 0.753 | 0.944 | **0.670** |
| **Transformer** | **0.867** | **0.879** | 0.801 | **0.768** | **0.948** | 0.614 |

Evaluated on a held-out test set of **15 subjects (19,166 epochs)** never seen during training, so these reflect generalization to unseen animals. The two sequence models lead: the **Transformer** has the best accuracy, balanced accuracy, κ, and AUC, while **CNN + LSTM** has the best macro-F1 and REM-F1. Cohen's κ ≈ 0.77 indicates substantial agreement with expert scoring. REM (only ~6% of epochs) reaches 0.86–0.92 recall through class weighting alone — no oversampling.

---

## Dataset

**OpenNeuro ds006366** — Rat sleep EEG/EMG recordings

- 92 rats, 148 recordings, ~128,250 labeled epochs
- 2 channels per recording: EEG1 and EMG
- Sampling rate: 128 Hz
- Epoch length: 4 seconds (512 samples)
- Class distribution: Wake 43% / NREM 50% / REM 7%

Download via [OpenNeuro](https://openneuro.org/datasets/ds006366) or DataLad:
```bash
datalad install https://github.com/OpenNeuroDatasets/ds006366.git
cd ds006366 && datalad get .
```

Place the downloaded folder as `ds006366/` in the project root.

---

## Project Structure

```
.
├── ds006366/                        # Raw EDF recordings (OpenNeuro dataset)
├── data/                            # Processed arrays (Git LFS)
│   ├── X_all.npy                    # Raw epochs (128250, 2, 512)
│   ├── F_all.npy                    # Feature vectors (128250, 33)
│   ├── y_all.npy                    # Labels 0=Wake 1=NREM 2=REM
│   ├── subjects_all.npy             # Subject IDs per epoch
│   ├── X_train/val/test.npy         # Raw signal splits
│   ├── F_train/val/test.npy         # Feature splits (33 features)
│   ├── class_weights.npy            # Inverse-frequency weights
│   ├── splits_metadata.json         # Train/val/test subject assignments
│   ├── training_histories.json      # Loss/accuracy curves per model
│   └── evaluation_results.json      # Final test metrics
├── checkpoints/                     # Saved model weights (Git LFS)
│   ├── mlp_best.pt / mlp_final.pt
│   ├── cnn_best.pt / cnn_final.pt
│   ├── cnn_lstm_best.pt / cnn_lstm_final.pt
│   └── transformer_best.pt / transformer_final.pt
├── figures/                         # All generated plots (22 figures)
├── src/
│   ├── config.py                    # Central path configuration
│   ├── Step1_Explore/               # Dataset exploration scripts
│   ├── Step2_Preprocess/            # EDF loading and preprocessing
│   ├── Step3_FeatureExtraction/     # 33-feature extraction pipeline
│   ├── Step4_DatasetSplitting/      # Subject-level train/val/test splits
│   ├── Step5_Model/                 # Model definitions, training loop
│   └── Step6_Evaluation/           # Metrics, plots, and evaluation
└── run_pipeline.py                  # Full pipeline runner
```

---

## Pipeline

Each step can be run independently or via `run_pipeline.py`:

```bash
python run_pipeline.py
```

### Step 1 — Exploration
```bash
python src/Step1_Explore/explore_structure.py
python src/Step1_Explore/explore_signals.py
python src/Step1_Explore/explore_epochs.py
python src/Step1_Explore/explore_crosssubject.py
```
Generates figures 01–10: dataset overview, raw signals, hypnograms, PSD per stage, band power heatmaps, per-subject distributions.

### Step 2 — Preprocessing
```bash
python src/Step2_Preprocess/preprocess.py
```
Pipeline per recording:
1. Load EDF (MNE)
2. FIR bandpass filter 0.5–45 Hz (Hamming window)
3. Select EEG1 + EMG channels
4. Artifact clipping at ±6σ
5. Global Z-score normalization — per channel, computed over the whole recording, applied *before* epoching (not per 4-s epoch)
6. Slice into 4-second epochs (512 samples)

Saves `X_all.npy` (128250, 2, 512) and `y_all.npy`.

### Step 3 — Feature Extraction
```bash
python src/Step3_FeatureExtraction/features.py
```
Extracts **33 features** per epoch using Welch's method:

| Group | Features | Count |
|---|---|---|
| EEG band powers | log absolute + relative power in δ/θ/α/σ/β/γ | 12 |
| EEG spectral | spectral edge (95%), spectral entropy, spectral centroid | 3 |
| EEG temporal | mean, std, variance, skewness, kurtosis, RMS, ZCR, line length | 8 |
| EEG Hjorth | activity, mobility, complexity | 3 |
| EMG | RMS, HF power (>30 Hz), HF ratio, variance | 4 |
| Cross-channel | θ/δ ratio, α/δ ratio, EEG-EMG cross-correlation | 3 |

Saves `F_all.npy` (128250, 33) and generates feature importance plots.

### Step 4 — Dataset Splitting
```bash
python src/Step4_DatasetSplitting/build_splits.py
```
- **Subject-level split** (no data leakage): 64 train / 13 val / 15 test subjects
- Computes inverse-frequency class weights for weighted cross-entropy loss
- Saves all split arrays and `splits_metadata.json`

### Step 5 — Training
```bash
python src/Step5_Model/train.py
```
Trains all four models sequentially on the same GPU. Key training details:
- Weighted cross-entropy loss with label smoothing (0.05)
- Adam optimizer with ReduceLROnPlateau scheduler (patience=8, factor=0.5)
- Early stopping on validation loss
- Mixed precision (AMP) for faster GPU training
- Gradient clipping (max norm = 1.0)
- **Data augmentation** for CNN/CNN+LSTM/Transformer: Gaussian noise (σ=0.02), amplitude jitter (×0.8–1.2), random time shift (±16 samples, zero-padded — not circular). Fully vectorized as batched GPU ops (no per-sample Python loop)

### Step 6 — Evaluation
```bash
python src/Step6_Evaluation/run_evaluation.py
```
Produces per-model: confusion matrices, ROC curves, per-subject accuracy bars, hypnogram comparisons, and cross-model comparison plots.

---

## Model Architectures

### Baseline MLP
Trained on the 33 handcrafted features.
```
Input(33) -> Linear(256) -> BN -> ReLU -> Dropout(0.35)
          -> Linear(128) -> BN -> ReLU -> Dropout(0.35)
          -> Linear(64)  -> BN -> ReLU -> Dropout(0.35)
          -> Linear(3)
```
Parameters: ~50K

### 1D CNN
Trained directly on raw EEG+EMG signals (2, 512). Four convolutional blocks with progressively smaller kernels to capture oscillations at multiple timescales.
```
Input(2, 512)
-> ConvBlock(2→32,   kernel=50, pool=4)   # slow waves / delta
-> ConvBlock(32→64,  kernel=25, pool=2)   # spindles / alpha
-> ConvBlock(64→128, kernel=10, pool=2)   # fast oscillations
-> ConvBlock(128→256, kernel=5, pool=2)
-> GlobalAvgPool -> Linear(256→128) -> Linear(128→3)
```
Parameters: ~335K

### CNN + LSTM (best model)
Processes sequences of 31 consecutive epochs (124 seconds of context). CNN encodes each epoch independently; BiLSTM models temporal transitions.
```
Input(batch, 31, 2, 512)
-> CNN encoder per epoch -> (batch, 31, 256)
-> BiLSTM(hidden=256, layers=2) -> (batch, 31, 512)
-> center output -> Linear(512→64) -> Linear(64→3)
```
Parameters: ~3.0M

### Transformer
Same CNN backbone as CNN+LSTM, but replaces the LSTM with a 4-layer Transformer encoder with sinusoidal positional encoding. Pre-norm (norm_first=True) for training stability.
```
Input(batch, 31, 2, 512)
-> CNN encoder per epoch -> (batch, 31, 256)
-> PositionalEncoding
-> TransformerEncoder(layers=4, heads=8, d_ff=512)
-> center output -> LayerNorm -> Linear(256→64) -> Linear(64→3)
```
Parameters: ~2.4M

---

## Setup

### Requirements
```bash
# Training requires a CUDA build of PyTorch — the trainer runs on the GPU
# (no CPU fallback). Pick the wheel matching your driver, e.g. CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install mne numpy scipy scikit-learn matplotlib tqdm
```

### Running on a new machine
All paths are computed relative to `config.py` using `Path(__file__).resolve()` — no hardcoded paths.

```bash
git clone <repo>
cd <repo>
git lfs pull          # download .npy arrays and .pt checkpoints
pip install -r requirements.txt   # or install manually (see above)
python run_pipeline.py            # run full pipeline from scratch
```

To skip retraining and only run evaluation on the saved checkpoints:
```bash
python src/Step6_Evaluation/run_evaluation.py
```

---

## Key Design Decisions

**Class imbalance (REM = 6%):** Inverse-frequency class weights in the loss function only — no oversampling during training. For the sequence models (CNN+LSTM, Transformer) oversampling is especially harmful: duplicating/shuffling minority epochs into the input stream would scramble the natural sleep-stage transitions the recurrent/attention layers rely on. Class weights `[0.32, 0.29, 2.39]` handle the imbalance while leaving raw epoch order intact.

**Subject-level splits:** All epochs from the same rat stay in the same split. This prevents data leakage and gives a realistic estimate of generalization to new animals.

**Global normalization:** Z-score per channel computed over the entire recording and applied before epoching — not per 4-second epoch. This puts each channel on a comparable scale while preserving the relative amplitude differences *between* epochs (e.g. the drop in EMG amplitude during REM) that help separate the stages; per-epoch normalization would erase that information.

**Temporal context (31 epochs = 124s):** Sleep stages have strong temporal autocorrelation — REM always follows NREM, Wake-to-sleep transitions are gradual. The LSTM and Transformer exploit this structure; the standalone CNN cannot.

---

## Figures

| Figure | Description |
|---|---|
| 01–08 | Dataset exploration: signals, hypnograms, PSDs, band powers |
| 09–10 | Preprocessed epoch examples and amplitude distribution |
| 11–13 | Feature importance (Fisher score), top feature distributions, correlation matrix |
| 14–15 | Train/val/test split distribution and subject assignment |
| 16 | Training curves (loss and accuracy) for all models |
| 17 | Confusion matrices (normalized + raw counts) per model |
| 18 | ROC curves (one-vs-rest) per model |
| 19 | Per-subject accuracy, macro F1, and REM F1 on test subjects |
| 20 | True vs predicted hypnogram for a test subject |
| 21 | Model comparison bar chart (accuracy, balanced acc, F1, kappa) |
| 22 | REM classification deep-dive: precision, recall, F1 per model |

---

## License

See [LICENSE](LICENSE).
