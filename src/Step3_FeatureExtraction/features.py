# src/features.py
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # project root
sys.path.insert(0, str(_HERE))

import numpy as np
from scipy.signal import welch
from scipy.stats  import skew, kurtosis
from tqdm import tqdm
from src.config import DATA_DIR, SFREQ

# ── Frequency bands (Hz) ──────────────────────────────────────────────────────
FREQ_BANDS = {
    "delta":  (0.5,  4.0),
    "theta":  (4.0,  8.0),
    "alpha":  (8.0, 13.0),
    "sigma":  (12.0, 16.0),   # sleep spindles
    "beta":   (16.0, 30.0),
    "gamma":  (30.0, 45.0),
}

SFREQ     = 128
NPERSEG   = 256   # Welch window — 2 seconds, good freq resolution at 128 Hz


# ══════════════════════════════════════════════════════════════════════════════
#  SPECTRAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_band_powers(epoch_1d: np.ndarray) -> dict:
    """
    Compute absolute and relative power in each frequency band
    using Welch's method.

    Welch splits the signal into overlapping windows, computes the
    FFT of each, then averages — much more stable than a single FFT
    on a 4-second window.

    Input : 1D array (512,)
    Output: dict with 12 values (6 bands × absolute + relative)
    """
    freqs, psd = welch(epoch_1d, fs=SFREQ, nperseg=NPERSEG)

    total_power = psd.sum() + 1e-12
    features    = {}

    for band, (lo, hi) in FREQ_BANDS.items():
        mask    = (freqs >= lo) & (freqs < hi)
        abs_pow = psd[mask].sum()
        features[f"{band}_abs"] = float(np.log10(abs_pow + 1e-12))  # log scale
        features[f"{band}_rel"] = float(abs_pow / total_power)       # ratio

    return features


def compute_spectral_edge(epoch_1d: np.ndarray, edge: float = 0.95) -> float:
    """
    Spectral edge frequency — the frequency below which `edge`
    fraction of total power sits. High in Wake (fast activity),
    low in NREM (slow delta dominates).
    """
    freqs, psd = welch(epoch_1d, fs=SFREQ, nperseg=NPERSEG)
    cumulative = np.cumsum(psd)
    threshold  = edge * cumulative[-1]
    idx        = np.searchsorted(cumulative, threshold)
    return float(freqs[min(idx, len(freqs) - 1)])


def compute_spectral_entropy(epoch_1d: np.ndarray) -> float:
    """
    Spectral entropy — measures how spread out the power spectrum is.
    High entropy = power spread across many frequencies (Wake).
    Low entropy  = power concentrated in a few bands (NREM delta peak).
    """
    _, psd = welch(epoch_1d, fs=SFREQ, nperseg=NPERSEG)
    psd_norm = psd / (psd.sum() + 1e-12)
    entropy  = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return float(entropy)


def compute_spectral_centroid(epoch_1d: np.ndarray) -> float:
    """
    Power-weighted mean frequency (spectral centroid).
    NREM: ~2-4 Hz (delta dominant), REM: ~5-8 Hz (theta), Wake: ~8-15 Hz.
    Robust to per-epoch Z-score normalization (global scaling cancels).
    """
    freqs, psd = welch(epoch_1d, fs=SFREQ, nperseg=NPERSEG)
    total = psd.sum() + 1e-12
    return float(np.sum(freqs * psd) / total)


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_temporal_features(epoch_1d: np.ndarray) -> dict:
    """
    Statistical descriptors of the raw waveform.
    These capture amplitude, variability, and shape.

    Input : 1D array (512,)
    Output: dict with 8 values
    """
    return {
        "mean":            float(np.mean(epoch_1d)),
        "std":             float(np.std(epoch_1d)),
        "variance":        float(np.var(epoch_1d)),
        "skewness":        float(skew(epoch_1d)),
        "kurtosis":        float(kurtosis(epoch_1d)),
        "rms":             float(np.sqrt(np.mean(epoch_1d ** 2))),
        "zero_cross_rate": float(
            ((epoch_1d[:-1] * epoch_1d[1:]) < 0).sum() / len(epoch_1d)
        ),
        "line_length":     float(np.sum(np.abs(np.diff(epoch_1d)))),
    }


def compute_hjorth_params(epoch_1d: np.ndarray) -> dict:
    """
    Hjorth parameters — classic EEG descriptors from 1970, still useful.

    Activity   = variance of the signal         (overall power)
    Mobility   = std of 1st derivative / std    (mean frequency proxy)
    Complexity = mobility of derivative / mobility (frequency change rate)

    NREM: low mobility (slow waves), low complexity
    Wake: high mobility + complexity (fast irregular signal)
    """
    d1 = np.diff(epoch_1d)
    d2 = np.diff(d1)

    var0 = np.var(epoch_1d) + 1e-12
    var1 = np.var(d1)       + 1e-12
    var2 = np.var(d2)       + 1e-12

    activity   = float(var0)
    mobility   = float(np.sqrt(var1 / var0))
    complexity = float(np.sqrt(var2 / var1) / mobility)

    return {
        "hjorth_activity":   activity,
        "hjorth_mobility":   mobility,
        "hjorth_complexity": complexity,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EMG-SPECIFIC FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_emg_features(emg_1d: np.ndarray) -> dict:
    """
    EMG features focused on muscle activity level.
    This is the key signal for separating REM from Wake —
    both look similar on EEG, but EMG is near-zero in REM.

    High-frequency power (> 30 Hz) is the most reliable EMG marker.
    """
    freqs, psd = welch(emg_1d, fs=SFREQ, nperseg=NPERSEG)

    hf_mask    = freqs >= 30
    hf_power   = float(np.log10(psd[hf_mask].sum() + 1e-12))
    total_power = psd.sum() + 1e-12

    return {
        "emg_rms":      float(np.sqrt(np.mean(emg_1d ** 2))),
        "emg_hf_power": hf_power,
        "emg_hf_ratio": float(psd[hf_mask].sum() / total_power),
        "emg_variance": float(np.var(emg_1d)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED FEATURE VECTOR PER EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def extract_epoch_features(epoch: np.ndarray) -> np.ndarray:
    """
    Extract all features from one epoch.

    Input : epoch of shape (2, 512)  — [EEG1, EMG]
    Output: 1D feature vector of shape (N_FEATURES,)

    Feature breakdown:
        EEG band powers           : 12  (6 bands × abs + rel)
        EEG spectral edge         :  1
        EEG spectral entropy      :  1
        EEG spectral centroid     :  1  (power-weighted mean frequency)
        EEG temporal stats        :  8
        EEG Hjorth params         :  3
        EMG features              :  4
        Cross-channel features    :  3  (theta/delta ratio, alpha/delta ratio, xcorr)
        ──────────────────────────────
        Total                     : 33
    """
    eeg = epoch[0]   # EEG1  — shape (512,)
    emg = epoch[1]   # EMG   — shape (512,)

    features = {}
    bp = compute_band_powers(eeg)
    features.update(bp)
    features["spectral_edge"]      = compute_spectral_edge(eeg)
    features["spectral_entropy"]   = compute_spectral_entropy(eeg)
    features["spectral_centroid"]  = compute_spectral_centroid(eeg)
    features.update(compute_temporal_features(eeg))
    features.update(compute_hjorth_params(eeg))
    emg_feats = compute_emg_features(emg)
    features.update(emg_feats)

    # Cross-channel features — valid after per-epoch Z-score normalization
    # (spectral ratios cancel global scaling; xcorr uses shape not amplitude)
    features["theta_delta_ratio"] = bp["theta_abs"] - bp["delta_abs"]
    features["alpha_delta_ratio"] = bp["alpha_abs"] - bp["delta_abs"]

    eeg_norm = (eeg - eeg.mean()) / (eeg.std() + 1e-12)
    emg_norm = (emg - emg.mean()) / (emg.std() + 1e-12)
    features["eeg_emg_xcorr"] = float(np.mean(eeg_norm * emg_norm))

    return np.array(list(features.values()), dtype=np.float32)


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names matching extract_epoch_features."""
    names = []
    for band in FREQ_BANDS:
        names += [f"eeg_{band}_abs", f"eeg_{band}_rel"]
    names += [
        "eeg_spectral_edge", "eeg_spectral_entropy", "eeg_spectral_centroid",
        "eeg_mean", "eeg_std", "eeg_variance",
        "eeg_skewness", "eeg_kurtosis", "eeg_rms",
        "eeg_zero_cross_rate", "eeg_line_length",
        "eeg_hjorth_activity", "eeg_hjorth_mobility", "eeg_hjorth_complexity",
        "emg_rms", "emg_hf_power", "emg_hf_ratio", "emg_variance",
        "theta_delta_ratio", "alpha_delta_ratio", "eeg_emg_xcorr",
    ]
    return names


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH EXTRACTION — run over all epochs
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(X: np.ndarray) -> np.ndarray:
    """
    Extract features from every epoch in the dataset.

    Input : X of shape (N, 2, 512)
    Output: F of shape (N, 33)
    """
    features = []
    for epoch in tqdm(X, desc="Extracting features", unit="epoch"):
        features.append(extract_epoch_features(epoch))
    return np.stack(features, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — run this file directly to build the feature matrix
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.config import FIGURES_DIR

    print("Loading preprocessed epochs...")
    X = np.load(DATA_DIR / "X_all.npy")
    y = np.load(DATA_DIR / "y_all.npy")
    print(f"  X shape: {X.shape}")

    feature_names = get_feature_names()
    print(f"  Feature count: {len(feature_names)}")
    print(f"  Features: {feature_names}\n")

    print("Extracting features (this takes a few minutes)...")
    F = extract_all_features(X)

    print(f"\nFeature matrix shape: {F.shape}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\nSanity checks:")
    if np.isnan(F).any():
        print("  FAIL: NaN values found")
    else:
        print("  PASS: no NaN values")
    if np.isinf(F).any():
        print("  FAIL: Inf values found")
    else:
        print("  PASS: no Inf values")
    print(f"  Value range: [{F.min():.3f}, {F.max():.3f}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(DATA_DIR / "F_all.npy", F)
    print(f"\nSaved -> {DATA_DIR / 'F_all.npy'}  ({F.nbytes / 1e6:.1f} MB)")

    # ── Plot 1: Feature importance via class separability ─────────────────────
    # For each feature, compute how well it separates the 3 classes
    # using the ratio of between-class to within-class variance (Fisher score)
    print("\nComputing feature separability scores...")

    class_names = {0: "Wake", 1: "NREM", 2: "REM"}
    colors      = {0: "#378ADD", 1: "#1D9E75", 2: "#EF9F27"}

    overall_mean = F.mean(axis=0)
    fisher_scores = []
    for fi in range(F.shape[1]):
        between = sum(
            (y == c).sum() * (F[y == c, fi].mean() - overall_mean[fi]) ** 2
            for c in [0, 1, 2]
        )
        within = sum(
            F[y == c, fi].var() * (y == c).sum()
            for c in [0, 1, 2]
        )
        fisher_scores.append(between / (within + 1e-12))

    fisher_scores = np.array(fisher_scores)
    sorted_idx    = np.argsort(fisher_scores)[::-1]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(feature_names)),
           fisher_scores[sorted_idx],
           color="#7F77DD", edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(
        [feature_names[i] for i in sorted_idx],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Fisher separability score")
    ax.set_title("Feature importance — class separability (higher = better)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "11_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")

    # ── Plot 2: Top 6 features — distribution per class ───────────────────────
    top6_idx = sorted_idx[:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for plot_i, feat_i in enumerate(top6_idx):
        ax = axes[plot_i]
        for cls in [0, 1, 2]:
            vals = F[y == cls, feat_i]
            ax.hist(vals, bins=60, alpha=0.5,
                    color=colors[cls], label=class_names[cls], density=True)
        ax.set_title(feature_names[feat_i], fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Top 6 most discriminative features", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "12_top_features_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")

    # ── Plot 3: Feature correlation heatmap ───────────────────────────────────
    print("Computing correlation matrix...")
    corr = np.corrcoef(F.T)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title("Feature correlation matrix")
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    out = FIGURES_DIR / "13_feature_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nTop 10 most discriminative features:")
    for rank, i in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:>2}. {feature_names[i]:<35} score={fisher_scores[i]:.2f}")
