"""
Full pipeline runner — re-extracts features and retrains all models.
Run this from the project root:
    python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

steps = [
    ("Feature extraction (33 features)",
     ROOT / "src" / "Step3_FeatureExtraction" / "features.py"),
    ("Dataset splitting",
     ROOT / "src" / "Step4_DatasetSplitting" / "build_splits.py"),
    ("Model training (MLP + CNN + CNN+LSTM + Transformer)",
     ROOT / "src" / "Step5_Model" / "train.py"),
    ("Evaluation",
     ROOT / "src" / "Step6_Evaluation" / "run_evaluation.py"),
]

for label, script in steps:
    print(f"\n{'='*65}")
    print(f"  STEP: {label}")
    print(f"{'='*65}")
    result = subprocess.run([PYTHON, str(script)], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\nFAILED at step: {label}")
        sys.exit(result.returncode)

print("\nAll steps completed successfully.")
