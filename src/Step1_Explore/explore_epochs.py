# src/explore_epochs.py
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

BASE   = Path("../../ds006366")
SUB_ID = 1
RUN_ID = 1

SFREQ     = 128
EPOCH_LEN = 4           # seconds
N_SAMPLES = SFREQ * EPOCH_LEN   # 512

STAGE_MAP    = {1: "Wake", 2: "NREM", 3: "REM"}
STAGE_COLORS = {1: "#378ADD", 2: "#1D9E75", 3: "#EF9F27"}

FREQ_BANDS = {
    "Delta":  (0.5, 4),
    "Theta":  (4,   8),
    "Alpha":  (8,  13),
    "Sigma":  (12, 16),
    "Beta":   (16, 30),
    "Gamma":  (30, 45),
}

# ── Load ──────────────────────────────────────────────────────────────────────
sub  = f"sub-{SUB_ID:03d}"
raw  = mne.io.read_raw_edf(
    str(BASE / sub / "eeg" / f"{sub}_task-sleep_run-{RUN_ID}_eeg.edf"),
    preload=True, verbose=False
)
raw.filter(0.5, 45.0, method="fir", fir_window="hamming", verbose=False)
events = pd.read_csv(
    BASE / sub / "eeg" / f"{sub}_task-sleep_run-{RUN_ID}_events.tsv", sep="\t"
)
data, _ = raw[:]    # (n_channels, n_total_samples)

# ── Extract one EEG channel's epochs per stage ───────────────────────────────
eeg_ch = 0   # EEG1 is always channel 0

stage_epochs = {1: [], 2: [], 3: []}

for _, row in events.iterrows():
    s = int(row["stage"])
    if s not in stage_epochs:
        continue
    start = int(row["onset"] * SFREQ)
    end   = start + N_SAMPLES
    if end > data.shape[1]:
        continue
    stage_epochs[s].append(data[eeg_ch, start:end] * 1e6)   # convert to µV

for s in stage_epochs:
    stage_epochs[s] = np.array(stage_epochs[s])
    print(f"{STAGE_MAP[s]:<6}: {len(stage_epochs[s])} epochs")

# ── Plot 1: Example epochs per stage ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
t = np.linspace(0, EPOCH_LEN, N_SAMPLES)

for ax, (s, name) in zip(axes, STAGE_MAP.items()):
    epochs = stage_epochs[s]
    # plot 5 example epochs semi-transparently
    for epoch in epochs[:5]:
        ax.plot(t, epoch, color=STAGE_COLORS[s], alpha=0.4, linewidth=0.8)
    # plot the mean on top
    ax.plot(t, epochs[:50].mean(axis=0),
            color=STAGE_COLORS[s], linewidth=1.8, label="Mean of 50 epochs")
    ax.set_ylabel(f"{name}\n(µV)", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time (s)")
plt.suptitle(f"sub-{SUB_ID:03d} — Example EEG epochs per sleep stage", fontsize=12)
plt.tight_layout()
plt.savefig("../../figures/04_epoch_examples.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Plot 2: Power spectral density per stage ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

for s, name in STAGE_MAP.items():
    epochs  = stage_epochs[s]
    psds    = []
    for epoch in epochs[:200]:   # use up to 200 epochs
        f, psd = welch(epoch, fs=SFREQ, nperseg=SFREQ*2)
        psds.append(psd)
    mean_psd = np.mean(psds, axis=0)

    # Plot only up to 45 Hz
    mask = f <= 45
    ax.semilogy(f[mask], mean_psd[mask],
                color=STAGE_COLORS[s], linewidth=2, label=name)

# Mark frequency bands
band_colors = ["#EEEDFE", "#E1F5EE", "#FAEEDA", "#FAECE7", "#E6F1FB", "#EAF3DE"]
for (band, (lo, hi)), bc in zip(FREQ_BANDS.items(), band_colors):
    ax.axvspan(lo, hi, alpha=0.25, color=bc, label=band)
    ax.text((lo + hi) / 2, ax.get_ylim()[0] * 1.5, band,
            ha="center", fontsize=8, color="#444")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (µV²/Hz)")
ax.set_title(f"sub-{SUB_ID:03d} — Power spectral density per stage (EEG1)", fontsize=12)
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(True, alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("../../figures/05_psd_per_stage.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Plot 3: Band power heatmap ────────────────────────────────────────────────
band_power = {name: [] for name in STAGE_MAP.values()}
for s, name in STAGE_MAP.items():
    for epoch in stage_epochs[s][:200]:
        f, psd = welch(epoch, fs=SFREQ, nperseg=SFREQ*2)
        row = []
        for band, (lo, hi) in FREQ_BANDS.items():
            mask = (f >= lo) & (f <= hi)
            row.append(np.log10(psd[mask].mean() + 1e-12))
        band_power[name].append(row)

matrix = np.array([np.mean(band_power[name], axis=0)
                   for name in STAGE_MAP.values()])

fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(FREQ_BANDS)))
ax.set_xticklabels(FREQ_BANDS.keys(), fontsize=10)
ax.set_yticks(range(len(STAGE_MAP)))
ax.set_yticklabels(STAGE_MAP.values(), fontsize=10)
ax.set_title("Band power heatmap by sleep stage (log scale)", fontsize=12)
plt.colorbar(im, ax=ax, label="Log power (µV²/Hz)")

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, f"{matrix[i,j]:.2f}",
                ha="center", va="center", fontsize=9, color="black")

plt.tight_layout()
plt.savefig("../../figures/06_band_power_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved -> figures/04_epoch_examples.png")
print("Saved -> figures/05_psd_per_stage.png")
print("Saved -> figures/06_band_power_heatmap.png")