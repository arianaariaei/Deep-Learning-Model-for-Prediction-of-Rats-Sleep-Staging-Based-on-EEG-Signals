# src/config.py
from pathlib import Path

# ── Edit these paths to match your machine ────────────────────────────────────
BASE         = Path(r"D:\university\4042\Deep-Learning-Model-for-Prediction-of-Rats-Sleep-Staging-Based-on-EEG-Signals\ds006366")
FIGURES_DIR  = Path(r"D:\university\4042\Deep-Learning-Model-for-Prediction-of-Rats-Sleep-Staging-Based-on-EEG-Signals\figures")
DATA_DIR     = Path(r"D:\university\4042\Deep-Learning-Model-for-Prediction-of-Rats-Sleep-Staging-Based-on-EEG-Signals\data")
# ─────────────────────────────────────────────────────────────────────────────

# Signal constants
SFREQ      = 128
EPOCH_LEN  = 4
N_SAMPLES  = SFREQ * EPOCH_LEN   # 512

# Sleep stage mapping (raw code → human label)
STAGE_MAP = {1: "Wake", 2: "NREM", 3: "REM", 4: "Artifact"}

# Create output folders if they don't exist
for d in [FIGURES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Validate dataset path on import
if not BASE.exists():
    raise FileNotFoundError(
        f"\nDataset folder not found: {BASE}\n"
        f"Please update the BASE path in src/config.py"
    )

event_files = list(BASE.glob("sub-*/eeg/*_events.tsv"))
if not event_files:
    raise FileNotFoundError(
        f"\nNo events.tsv files found under: {BASE}\n"
        f"Please check your folder structure."
    )

print(f"[config] Dataset OK — {len(event_files)} recordings found.")