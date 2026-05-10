# src/build_dataset.py
"""
Run this script once to preprocess all subjects and save the results
as numpy arrays. Subsequent steps (features, model) load from disk
instead of re-running the pipeline every time.

Output files (saved to data/):
    X_all.npy          — (N, C, 512)  float32 epochs
    y_all.npy          — (N,)         int64   labels
    subjects_all.npy   — (N,)         int64   subject ID per epoch
    runs_all.npy       — (N,)         int64   run ID per epoch
    metadata.json      — summary stats
"""

import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from src.config import BASE, DATA_DIR
from preprocess import preprocess_subject

# ── Settings ─────────────────────────────────────────────────────────────────
# Set TMAX_SECONDS to None to process full recordings.
# Use a small value (e.g. 3600) while testing to keep things fast.
TMAX_SECONDS = 3600   # process first hour per subject; set None for full dataset

# Which subjects to include. Adjust based on which EDFs you've downloaded.
participants = pd.read_csv(BASE / "participants.tsv", sep="\t")
ALL_SUBJECT_IDS = [
    int(row["participant_id"].replace("sub-", ""))
    for _, row in participants.iterrows()
]

print(f"Total subjects found: {len(ALL_SUBJECT_IDS)}")
print(f"Processing first {TMAX_SECONDS}s per subject (set TMAX_SECONDS=None for full)\n")

# ── Run pipeline ──────────────────────────────────────────────────────────────
X_list, y_list, sub_list, run_list = [], [], [], []
subject_metadata = []
failed = []

for sid in tqdm(ALL_SUBJECT_IDS, desc="Subjects"):

    # Find all runs for this subject
    sub = f"sub-{sid:03d}"
    edf_files = sorted((BASE / sub / "eeg").glob("*_eeg.edf"))

    for edf in edf_files:
        # Skip stubs (files < 1KB are git-annex pointers, not real data)
        if edf.stat().st_size < 1000:
            print(f"  [{sub}] Skipping stub: {edf.name}")
            continue

        # Extract run number from filename
        run_str = [p for p in edf.stem.split("_") if p.startswith("run-")]
        run_id  = int(run_str[0].replace("run-", "")) if run_str else 1

        try:
            X, y, info = preprocess_subject(
                sub_id  = sid,
                run_id  = run_id,
                tmax    = TMAX_SECONDS,
                base_dir= BASE,
            )
            X_list.append(X)
            y_list.append(y)
            sub_list.append(np.full(len(y), sid,    dtype=np.int64))
            run_list.append(np.full(len(y), run_id, dtype=np.int64))
            subject_metadata.append(info)

        except Exception as e:
            print(f"\n  [{sub} run-{run_id}] ERROR: {e}")
            failed.append((sid, run_id, str(e)))

# ── Concatenate and save ──────────────────────────────────────────────────────
if not X_list:
    raise RuntimeError("No subjects were successfully processed. Check your EDF files.")

X_all    = np.concatenate(X_list,   axis=0)
y_all    = np.concatenate(y_list,   axis=0)
subs_all = np.concatenate(sub_list, axis=0)
runs_all = np.concatenate(run_list, axis=0)

DATA_DIR.mkdir(parents=True, exist_ok=True)
np.save(DATA_DIR / "X_all.npy",        X_all)
np.save(DATA_DIR / "y_all.npy",        y_all)
np.save(DATA_DIR / "subjects_all.npy", subs_all)
np.save(DATA_DIR / "runs_all.npy",     runs_all)

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(y_all)
print(f"\n{'='*55}")
print(f"  PREPROCESSING COMPLETE")
print(f"{'='*55}")
print(f"  Subjects processed : {len(subject_metadata)}")
print(f"  Failed             : {len(failed)}")
print(f"  Total epochs       : {total:,}")
print(f"  X shape            : {X_all.shape}")
print(f"  X dtype            : {X_all.dtype}")
print(f"  y dtype            : {y_all.dtype}")
print(f"\n  Class distribution:")
class_names = {0: "Wake", 1: "NREM", 2: "REM"}
for cls, name in class_names.items():
    n   = int((y_all == cls).sum())
    pct = n / total * 100
    print(f"    {name:<6}: {n:>8,}  ({pct:5.1f}%)")

# Save metadata as JSON
metadata = {
    "n_subjects":   len(subject_metadata),
    "n_epochs":     total,
    "shape":        list(X_all.shape),
    "tmax_seconds": TMAX_SECONDS,
    "class_counts": {
        class_names[cls]: int((y_all == cls).sum())
        for cls in [0, 1, 2]
    },
    "failed": failed,
    "subjects": subject_metadata,
}
with open(DATA_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n  Saved to: {DATA_DIR}")
print(f"    X_all.npy        {X_all.nbytes  / 1e9:.2f} GB")
print(f"    y_all.npy        {y_all.nbytes  / 1e6:.1f} MB")
print(f"    subjects_all.npy {subs_all.nbytes/ 1e6:.1f} MB")
print(f"    metadata.json")

if failed:
    print(f"\n  Failed subjects/runs:")
    for sid, rid, err in failed:
        print(f"    sub-{sid:03d} run-{rid}: {err}")