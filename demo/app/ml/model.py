# demo/app/ml/model.py
from pathlib import Path
import joblib

MODEL_PKL = Path(__file__).resolve().parent / "models" / "risk_xgboost.pkl"

# Cargar modelo entrenado
risk_model = joblib.load(MODEL_PKL)
