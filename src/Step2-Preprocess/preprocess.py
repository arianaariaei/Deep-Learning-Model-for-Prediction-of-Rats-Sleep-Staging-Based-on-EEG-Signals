# src/preprocess.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mne
import numpy as np
import pandas as pd
from src.config import BASE, SFREQ, EPOCH_LEN, N_SAMPLES, STAGE_MAP

L_FREQ     = 0.5
H_FREQ     = 45.0
CLIP_STD   = 6.0
LABEL_MAP  = {1: 0, 2: 1, 3: 2}

# ── These two channels exist in ALL labs ──────────────────────────────────────
COMMON_CHANNELS = ["EEG1", "EMG"]


def load_raw(edf_path: Path, tmax: float = None) -> mne.io.Raw:
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    if tmax is not None:
        tmax = min(tmax, raw.times[-1])
        raw.crop(tmin=0, tmax=tmax)
    raw.load_data(verbose=False)
    return raw


def bandpass_filter(raw: mne.io.Raw) -> mne.io.Raw:
    raw = raw.copy()
    raw.filter(
        l_freq     = L_FREQ,
        h_freq     = H_FREQ,
        method     = "fir",
        fir_window = "hamming",
        verbose    = False
    )
    return raw


def select_common_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Keep only EEG1 and EMG — the two channels present in every lab.
    This ensures all subjects produce arrays of the same shape (2, T).
    """
    available = raw.ch_names
    to_keep   = [ch for ch in COMMON_CHANNELS if ch in available]

    missing = [ch for ch in COMMON_CHANNELS if ch not in available]
    if missing:
        raise ValueError(
            f"Required channels {missing} not found in recording. "
            f"Available: {available}"
        )

    raw = raw.copy().pick_channels(to_keep, ordered=True)
    return raw


def clip_artifacts(data: np.ndarray, n_std: float = CLIP_STD) -> np.ndarray:
    data = data.copy()
    for ch in range(data.shape[0]):
        std = data[ch].std()
        if std < 1e-12:
            continue
        limit = n_std * std
        data[ch] = np.clip(data[ch], -limit, limit)
    return data


def slice_epochs(
    data:      np.ndarray,
    events_df: pd.DataFrame,
    sfreq:     float = SFREQ,
    epoch_len: int   = EPOCH_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(sfreq * epoch_len)
    X_list, y_list = [], []

    for _, row in events_df.iterrows():
        stage = int(row["stage"])
        if stage not in LABEL_MAP:
            continue

        onset = int(row["onset"] * sfreq)
        end   = onset + n_samples

        if end > data.shape[1]:
            continue

        X_list.append(data[:, onset:end])
        y_list.append(LABEL_MAP[stage])

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def normalize_epochs(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=2, keepdims=True)
    std  = X.std(axis=2,  keepdims=True)
    std  = np.where(std < 1e-10, 1.0, std)
    return ((X - mean) / std).astype(np.float32)


def preprocess_subject(
    sub_id:   int,
    run_id:   int   = 1,
    tmax:     float = None,
    base_dir: Path  = BASE,
) -> tuple[np.ndarray, np.ndarray, dict]:

    sub = f"sub-{sub_id:03d}"
    edf = base_dir / sub / "eeg" / f"{sub}_task-sleep_run-{run_id}_eeg.edf"
    tsv = base_dir / sub / "eeg" / f"{sub}_task-sleep_run-{run_id}_events.tsv"

    if not edf.exists():
        raise FileNotFoundError(f"EDF not found: {edf}")

    print(f"  [{sub} run-{run_id}] Loading...", end=" ", flush=True)
    raw    = load_raw(edf, tmax=tmax)
    events = pd.read_csv(tsv, sep="\t")

    print("Filtering...", end=" ", flush=True)
    raw = bandpass_filter(raw)

    # ── Select only EEG1 + EMG before anything else ───────────────────────────
    print("Selecting channels...", end=" ", flush=True)
    raw = select_common_channels(raw)

    print("Clipping...", end=" ", flush=True)
    data, _ = raw[:]
    data = clip_artifacts(data)

    print("Epoching...", end=" ", flush=True)
    X, y = slice_epochs(data, events)

    print("Normalizing...", end=" ", flush=True)
    X = normalize_epochs(X)

    counts = {
        "Wake": int((y == 0).sum()),
        "NREM": int((y == 1).sum()),
        "REM":  int((y == 2).sum()),
    }
    print(f"Done — {len(y)} epochs {counts}")

    info = {
        "sub_id":    sub_id,
        "run_id":    run_id,
        "n_epochs":  len(y),
        "n_channels": X.shape[1],
        "channels":  COMMON_CHANNELS,
        "counts":    counts,
    }
    return X, y, info