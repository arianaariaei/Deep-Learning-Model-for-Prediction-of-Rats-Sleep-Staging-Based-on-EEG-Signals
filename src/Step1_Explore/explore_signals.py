# src/explore_signals.py
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE        = Path(r"/ds006366")
FIGURES_DIR = Path(r"/figures")
FIGURES_DIR.mkdir(exist_ok=True)

SUB_ID = 1
RUN_ID = 1
SFREQ  = 128

STAGE_MAP    = {1: "Wake", 2: "NREM", 3: "REM", 4: "Artifact"}
STAGE_COLORS = {1: "#378ADD", 2: "#1D9E75", 3: "#EF9F27", 4: "#E24B4A"}

sub = f"sub-{SUB_ID:03d}"
edf = BASE / sub / "eeg" / f"{sub}_task-sleep_run-{RUN_ID}_eeg.edf"
tsv = BASE / sub / "eeg" / f"{sub}_task-sleep_run-{RUN_ID}_events.tsv"

# ── Load only first 10 minutes ────────────────────────────────────────────────
LOAD_SECONDS = 600
print(f"Loading first {LOAD_SECONDS//60} min from {edf.name} ...")

raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
print(f"Subject      : {sub}")
print(f"Channels     : {raw.ch_names}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Total samples: {raw.n_times:,}")
print(f"Duration     : {raw.times[-1]/3600:.2f} h")

events = pd.read_csv(tsv, sep="\t")
print(f"Total epochs : {len(events)}")

raw.crop(tmin=0, tmax=LOAD_SECONDS)
raw.load_data(verbose=False)
print(f"Cropped to {LOAD_SECONDS}s — shape: {raw.get_data().shape}")

# ── Plot 1: Raw signals ───────────────────────────────────────────────────────
print("Plotting raw signals...")
PLOT_SECONDS = 300
n_samp = PLOT_SECONDS * int(raw.info["sfreq"])
data, _ = raw[:, :n_samp]
times    = raw.times[:n_samp]

fig, axes = plt.subplots(len(raw.ch_names), 1,
                         figsize=(18, 2.8 * len(raw.ch_names)), sharex=True)
if len(raw.ch_names) == 1:
    axes = [axes]

for ax, ch_name, ch_data in zip(axes, raw.ch_names, data):
    scale = 1e6 if "EEG" in ch_name.upper() else 1e3
    unit  = "µV"  if "EEG" in ch_name.upper() else "mV"
    ax.plot(times, ch_data * scale, color="#2c2c2a", linewidth=0.4, alpha=0.85)
    ax.set_ylabel(f"{ch_name}\n({unit})", fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    for _, row in events[events["onset"] < PLOT_SECONDS].iterrows():
        end = min(row["onset"] + row["duration"], PLOT_SECONDS)
        ax.axvspan(row["onset"], end,
                   alpha=0.18, color=STAGE_COLORS.get(row["stage"], "#aaa"),
                   linewidth=0)

patches = [mpatches.Patch(color=STAGE_COLORS[s], alpha=0.6, label=STAGE_MAP[s])
           for s in [1, 2, 3, 4]]
axes[0].legend(handles=patches, loc="upper right", fontsize=8, ncol=4)
axes[-1].set_xlabel("Time (s)")
plt.suptitle(f"{sub} — Raw signals (first 5 min)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_raw_signals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> 02_raw_signals.png")

# ── Plot 2: Hypnogram — vectorized, no per-row loops ─────────────────────────
print("Plotting hypnogram...")

MAX_HOURS = 24
clean = events[(events["stage"] != 4) & (events["onset"] <= MAX_HOURS * 3600)].copy()

# Build a 1-sample-per-second signal encoding stage as y-value
# This is O(n_seconds) not O(n_epochs) — renders instantly
stage_y  = {1: 3, 2: 1, 3: 2}   # Wake=top, REM=mid, NREM=bottom
n_sec    = MAX_HOURS * 3600
hyp      = np.full(n_sec, np.nan)

for _, row in clean.iterrows():
    s     = int(row["onset"])
    e     = min(int(row["onset"] + row["duration"]), n_sec)
    hyp[s:e] = stage_y.get(int(row["stage"]), np.nan)

t_hours = np.arange(n_sec) / 3600

fig, ax = plt.subplots(figsize=(18, 3))

# One fill_between per stage — 3 calls total regardless of epoch count
for stage, y_val in stage_y.items():
    mask = (hyp == y_val)
    ax.fill_between(t_hours, y_val - 0.4, y_val + 0.4,
                    where=mask, color=STAGE_COLORS[stage],
                    linewidth=0, alpha=0.9)

ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["NREM", "REM", "Wake"])
ax.set_ylim(0.4, 3.6)
ax.set_xlim(0, MAX_HOURS)
ax.set_xlabel("Time (hours)")
ax.set_title(f"{sub} — Hypnogram (first {MAX_HOURS}h)", fontsize=12)
ax.grid(axis="x", alpha=0.3, linestyle="--")

patches = [mpatches.Patch(color=STAGE_COLORS[s], label=STAGE_MAP[s])
           for s in [1, 2, 3]]
ax.legend(handles=patches, loc="upper right", fontsize=9, ncol=3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_hypnogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> 03_hypnogram.png")

print("\nDone. Check your figures/ folder.")