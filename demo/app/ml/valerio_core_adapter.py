from pathlib import Path
import json, pickle
import pandas as pd

# --- rutas ---
THIS_DIR = Path(__file__).resolve().parent
VALERIO_ROOT = THIS_DIR.parents[2]              # .../VALERIO
CORE_ROOT = VALERIO_ROOT / "valerio_ai_project" # <- tu core

DATA_CSV = VALERIO_ROOT / "demo" / "app" / "data" / "data_finance.csv"
MODEL_PKL = VALERIO_ROOT / "demo" / "app" / "ml" / "models" / "risk_xgboost.pkl"
VECTORIZER_PKL = None
META_JSON = None
TARGET_COL = "target"   # <-- tu CSV tiene 'target', no 'risk'

_df = _model = _vec = None
_features = None

def load_core():
    global _df, _model, _vec, _features
    if _df is None:
        _df = pd.read_csv(DATA_CSV)
    if _model is None:
        with open(MODEL_PKL, "rb") as f:
            _model = pickle.load(f)
    if _vec is None and VECTORIZER_PKL and VECTORIZER_PKL.exists():
        with open(VECTORIZER_PKL, "rb") as f:
            _vec = pickle.load(f)
    if _features is None:
        if META_JSON and Path(META_JSON).exists():
            with open(META_JSON) as f:
                meta = json.load(f)
            _features = meta.get("features") or [c for c in _df.columns if c.lower() not in ("risk", "target")]
        else:
            # quitamos siempre la columna target/risk de las features
            _features = [c for c in _df.columns if c.lower() not in ("risk", "target")]
    return _df, _model, _vec, _features

def get_features(row_idx: int):
    df, *_ = load_core()
    return df[_features].iloc[row_idx:row_idx+1]

def predict_one(row_idx: int):
    df, model, vec, feats = load_core()
    X = df[feats].iloc[row_idx:row_idx+1]
    if vec is not None:
        X = vec.transform(X)
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(X)[0, 1])
        pred = int(p >= 0.5)
    else:
        pred = int(model.predict(X)[0])
        p = None
    y = int(df[TARGET_COL].iloc[row_idx]) if TARGET_COL in df.columns else None
    return {
        "row": row_idx,
        "prediction": pred,
        "probability": p,
        "actual": y
    }

# Wrappers accesibles por el agente
def ml_predict(row: int = 0):
    """Predicción accesible por el agente."""
    return predict_one(row)

def predict_by_row_index(row_idx: int = 0):
    """Alias para mantener compatibilidad con imports antiguos."""
    return predict_one(row_idx)

__all__ = ["ml_predict", "predict_one", "get_features", "predict_by_row_index"]
