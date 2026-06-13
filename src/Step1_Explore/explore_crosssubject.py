# src/explore_crosssubject.py
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

BASE = Path("../../ds006366")
SFREQ = 128
N_SUBJECTS_TO_PLOT = 10   # adjust based on how many EDFs you've downloaded

STAGE_MAP    = {1: "Wake", 2: "NREM", 3: "REM"}
STAGE_COLORS = {1: "#378ADD", 2: "#1D9E75", 3: "#EF9F27"}

# ── 1. Class distribution per subject ────────────────────────────────────────
event_files = sorted(BASE.glob("sub-*/eeg/*_events.tsv"))[:N_SUBJECTS_TO_PLOT]

rows = []
for f in event_files:
    sub = f.name.split("_")[0]
    df  = pd.read_csv(f, sep="\t")
    clean = df[df["stage"] != 4]
    total = len(clean)
    rows.append({
        "subject": sub,
        "Wake %":  (clean["stage"] == 1).sum() / total * 100,
        "NREM %":  (clean["stage"] == 2).sum() / total * 100,
        "REM %":   (clean["stage"] == 3).sum() / total * 100,
    })

dist_df = pd.DataFrame(rows).set_index("subject")

fig, ax = plt.subplots(figsize=(14, 5))
dist_df.plot(kind="bar", stacked=True, ax=ax,
             color=["#378ADD", "#1D9E75", "#EF9F27"],
             edgecolor="white", linewidth=0.3)
ax.set_ylabel("Percentage (%)")
ax.set_title("Sleep stage distribution per subject", fontsize=12)
ax.set_xticklabels(dist_df.index, rotation=45, ha="right", fontsize=8)
ax.legend(loc="upper right")
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig("../../figures/07_per_subject_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 2. Delta power comparison across subjects (requires EDFs) ─────────────────
print("\nComputing delta power per subject (requires downloaded EDF files)...")
delta_powers = {"Wake": [], "NREM": [], "REM": []}
subject_labels = []

for sub_dir in sorted(BASE.glob("sub-*"))[:N_SUBJECTS_TO_PLOT]:
    edf_files = list(sub_dir.glob("eeg/*.edf"))
    if not edf_files:
        continue
    edf = edf_files[0]
    if edf.stat().st_size < 1000:
        print(f"  {sub_dir.name}: stub, skipping")
        continue

    tsv = str(edf).replace("_eeg.edf", "_events.tsv")
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)
    events = pd.read_csv(tsv, sep="\t")
    data, _ = raw[:]
    subject_labels.append(sub_dir.name)

    for s, name in STAGE_MAP.items():
        powers = []
        for _, row in events[events["stage"] == s].head(50).iterrows():
            start = int(row["onset"] * SFREQ)
            end   = start + SFREQ * 4
            if end > data.shape[1]:
                continue
            epoch = data[0, start:end] * 1e6
            f, psd = welch(epoch, fs=SFREQ, nperseg=SFREQ)
            delta_mask = (f >= 0.5) & (f <= 4)
            powers.append(10 * np.log10(psd[delta_mask].mean() + 1e-12))
        delta_powers[name].append(np.mean(powers) if powers else np.nan)

if subject_labels:
    x = np.arange(len(subject_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, color) in enumerate(zip(["Wake", "NREM", "REM"],
                                           ["#378ADD", "#1D9E75", "#EF9F27"])):
        vals = delta_powers[name]
        ax.bar(x + i * width, vals, width, label=name,
               color=color, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x + width)
    ax.set_xticklabels(subject_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Delta power (dB)")
    ax.set_title("Delta band power per subject by sleep stage", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig("../../figures/08_delta_power_subjects.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved -> figures/08_delta_power_subjects.png")
else:
    print("  No real EDF files found — skipping delta power plot.")

print("Saved -> figures/07_per_subject_distribution.png")