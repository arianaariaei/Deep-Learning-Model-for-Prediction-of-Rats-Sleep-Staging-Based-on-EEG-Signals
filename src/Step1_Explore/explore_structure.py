# src/explore_structure.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path
from collections import defaultdict

BASE = Path("../../ds006366")

STAGE_MAP   = {1: "Wake", 2: "NREM", 3: "REM", 4: "Artifact"}
STAGE_COLORS = {1: "#378ADD", 2: "#1D9E75", 3: "#EF9F27", 4: "#E24B4A"}

# ── 1. Dataset-level metadata ──────────────────────────────────────────────────
with open(BASE / "task-sleep_eeg.json") as f:
    eeg_meta = json.load(f)

print("=" * 55)
print("  DATASET METADATA")
print("=" * 55)
for k, v in eeg_meta.items():
    print(f"  {k:<25} {v}")

# ── 2. Participants table ──────────────────────────────────────────────────────
parts = pd.read_csv(BASE / "participants.tsv", sep="\t")
print(f"\n  Subjects total : {len(parts)}")
print(f"  Labs           : {sorted(parts['lab'].unique())}")
print(f"\n  Subjects per lab:")
print(parts["lab"].value_counts().sort_index().to_string())

# ── 3. Lab channel configuration ──────────────────────────────────────────────
labs = pd.read_csv(BASE / "labs.tsv", sep="\t")
print(f"\n{'='*55}")
print("  CHANNEL CONFIG PER LAB")
print("=" * 55)
for _, row in labs.iterrows():
    print(f"  {row['lab']}: {row['channels']}")

# ── 4. Scan all event files ────────────────────────────────────────────────────
event_files = sorted(BASE.glob("sub-*/eeg/*_events.tsv"))
print(f"\n{'='*55}")
print(f"  LABEL STATISTICS  ({len(event_files)} recordings)")
print("=" * 55)

all_stages, rec_durations, sub_run_counts = [], [], defaultdict(int)

for f in event_files:
    sub = f.name.split("_")[0]
    df  = pd.read_csv(f, sep="\t")
    all_stages.append(df)
    duration_h = (df["onset"].max() + df["duration"].max()) / 3600
    rec_durations.append(duration_h)
    sub_run_counts[sub] += 1

all_df = pd.concat(all_stages, ignore_index=True)
total  = len(all_df)

print(f"  Total epochs     : {total:,}")
print(f"  Total duration   : {sum(rec_durations):.0f} h")
print(f"  Avg per recording: {np.mean(rec_durations):.1f} h")
print(f"\n  Stage distribution:")
for s in [1, 2, 3, 4]:
    n = (all_df["stage"] == s).sum()
    print(f"    {STAGE_MAP[s]:<10} {n:>8,}  ({n/total*100:5.1f}%)")

runs_series = pd.Series(sub_run_counts)
print(f"\n  Recordings per subject:")
print(runs_series.value_counts().sort_index().to_string())

# ── 5. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Dataset Overview", fontsize=13, fontweight="bold")

# (a) Class distribution — bar
counts = [int((all_df["stage"] == s).sum()) for s in [1, 2, 3, 4]]
labels = [STAGE_MAP[s] for s in [1, 2, 3, 4]]
colors = [STAGE_COLORS[s] for s in [1, 2, 3, 4]]
axes[0].bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
axes[0].set_title("Epoch count per stage")
axes[0].set_ylabel("Epochs")
for bar, val in zip(axes[0].patches, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                 f"{val/1e6:.2f}M", ha="center", va="bottom", fontsize=9)

# (b) Recording duration distribution
axes[1].hist(rec_durations, bins=20, color="#639922", edgecolor="white", linewidth=0.5)
axes[1].set_title("Recording duration per run")
axes[1].set_xlabel("Hours")
axes[1].set_ylabel("Count")
axes[1].axvline(np.mean(rec_durations), color="red", linestyle="--",
                linewidth=1, label=f"Mean {np.mean(rec_durations):.1f}h")
axes[1].legend(fontsize=9)

# (c) Subjects per lab
lab_counts = parts["lab"].value_counts().sort_index()
axes[2].bar(lab_counts.index, lab_counts.values, color="#7F77DD",
            edgecolor="white", linewidth=0.5)
axes[2].set_title("Subjects per lab")
axes[2].set_ylabel("Subjects")
for bar, val in zip(axes[2].patches, lab_counts.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("../../figures/01_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved -> figures/01_dataset_overview.png")