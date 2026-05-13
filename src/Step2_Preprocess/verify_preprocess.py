# src/verify_preprocessing.py
"""
Load the saved numpy arrays and run sanity checks.
Run this after build_dataset.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from src.config import DATA_DIR, FIGURES_DIR, STAGE_MAP

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading preprocessed data...")
X    = np.load(DATA_DIR / "X_all.npy")
y    = np.load(DATA_DIR / "y_all.npy")
subs = np.load(DATA_DIR / "subjects_all.npy")

with open(DATA_DIR / "metadata.json") as f:
    meta = json.load(f)

print(f"X shape : {X.shape}   (epochs, channels, samples)")
print(f"y shape : {y.shape}")
print(f"dtype   : {X.dtype}")
print(f"Subjects: {np.unique(subs).tolist()}")

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\nRunning sanity checks...")
errors = []

# 1. No NaN or Inf
if np.isnan(X).any():
    errors.append("FAIL: X contains NaN values")
else:
    print("  PASS: no NaN values")

if np.isinf(X).any():
    errors.append("FAIL: X contains Inf values")
else:
    print("  PASS: no Inf values")

# 2. Normalized — mean ≈ 0, std ≈ 1 per epoch per channel
means = X.mean(axis=2)    # (N, C)
stds  = X.std(axis=2)     # (N, C)
if abs(means.mean()) > 0.1:
    errors.append(f"FAIL: mean not near 0 (got {means.mean():.4f})")
else:
    print(f"  PASS: mean ≈ {means.mean():.4f}")
if abs(stds.mean() - 1.0) > 0.1:
    errors.append(f"FAIL: std not near 1 (got {stds.mean():.4f})")
else:
    print(f"  PASS: std  ≈ {stds.mean():.4f}")

# 3. Labels are only 0, 1, 2
unexpected = set(np.unique(y)) - {0, 1, 2}
if unexpected:
    errors.append(f"FAIL: unexpected labels found: {unexpected}")
else:
    print(f"  PASS: labels are {{0, 1, 2}} only")

# 4. Correct epoch length
if X.shape[2] != 512:
    errors.append(f"FAIL: expected 512 samples per epoch, got {X.shape[2]}")
else:
    print(f"  PASS: epoch length = {X.shape[2]} samples (4s × 128Hz)")

if errors:
    print("\nERRORS FOUND:")
    for e in errors:
        print(f"  {e}")
else:
    print("\nAll checks passed.")

# ── Plot: one example epoch per class ────────────────────────────────────────
class_names = {0: "Wake", 1: "NREM", 2: "REM"}
colors      = {0: "#378ADD", 1: "#1D9E75", 2: "#EF9F27"}
t           = np.linspace(0, 4, 512)

n_channels = X.shape[1]
fig, axes  = plt.subplots(3, n_channels, figsize=(5 * n_channels, 7),
                           sharex=True)

# If only one channel, axes is 1D — fix shape
if n_channels == 1:
    axes = axes.reshape(3, 1)

for cls in range(3):
    idx   = np.where(y == cls)[0]
    epoch = X[idx[0]]              # first epoch of this class (C, 512)

    for ch in range(n_channels):
        ax    = axes[cls, ch]
        label = "EEG" if ch < n_channels - 1 else "EMG"
        ax.plot(t, epoch[ch], color=colors[cls], linewidth=0.8)
        ax.set_ylabel(f"{class_names[cls]}\n{label}", fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        if cls == 0:
            ax.set_title(f"Channel {ch+1}: {label}", fontsize=10)

axes[-1, 0].set_xlabel("Time (s)")
plt.suptitle("Preprocessed epochs — one example per class", fontsize=12)
plt.tight_layout()
out = FIGURES_DIR / "09_preprocessed_epochs.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")

# ── Plot: amplitude distribution per class ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
eeg_ch  = 0   # plot EEG1

for cls in range(3):
    samples = X[y == cls, eeg_ch, :].flatten()
    # subsample for speed
    samples = samples[::10]
    ax.hist(samples, bins=100, alpha=0.5,
            color=colors[cls], label=class_names[cls], density=True)

ax.set_xlabel("Normalized amplitude (z-score)")
ax.set_ylabel("Density")
ax.set_title("EEG1 amplitude distribution after normalization")
ax.legend()
ax.set_xlim(-5, 5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
out = FIGURES_DIR / "10_amplitude_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")