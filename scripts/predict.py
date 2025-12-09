# scripts/predict.py

from pathlib import Path
import joblib
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
REPO = Path(__file__).resolve().parents[1]
DATA_IN = REPO / "data" / "raw" / "test.csv"    # change if needed
MODELS_DIR = REPO / "models"
OUT_DIR = REPO / "reports"
OUT_DIR.mkdir(exist_ok=True)

# --- add this, below REPO definition ---
import sys
# ensure Python can import 'src' as a package while running this script
sys.path.insert(0, str(REPO))
# --- end ---
# -----------------------------
# Load pipeline + model
# -----------------------------
pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")
model = joblib.load(MODELS_DIR / "rf.joblib")

# -----------------------------
# Load input data
# -----------------------------
df = pd.read_csv(DATA_IN)

if "Survived" in df.columns:
    X = df.drop(columns=["Survived"])
else:
    X = df.copy()

# -----------------------------
# Transform & predict
# -----------------------------
X_trans = pipeline.transform(X)
probs = model.predict_proba(X_trans)[:, 1]

# -----------------------------
# Save output
# -----------------------------
out = pd.DataFrame({
    "PassengerId": df.get("PassengerId", range(len(df))),
    "survived_proba": probs,
})

out_path = OUT_DIR / "predictions.csv"
out.to_csv(out_path, index=False)

print("Saved predictions →", out_path)