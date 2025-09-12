# app/ml/__init__.py
import pandas as pd
from pathlib import Path
import joblib

# ruta de los datos y modelo
DATA = Path(__file__).resolve().parents[1] / "data" / "data_finance.csv"
MODEL = Path(__file__).resolve().parent / "models" / "risk_xgboost.pkl"

# cargamos dataset
try:
    df = pd.read_csv(DATA)
except Exception as e:
    df = None
    print("⚠️ No se pudo cargar data_finance.csv:", e)

# cargamos modelo
try:
    risk_model = joblib.load(MODEL)
except Exception as e:
    risk_model = None
    print("⚠️ No se pudo cargar risk_xgboost.pkl:", e)

def ml_predict(row: int):
    """
    Devuelve la predicción para la fila 'row' del dataset financiero.
    """
    if df is None or risk_model is None:
        return {"error": "Modelo o dataset no cargado."}
    
    if row < 0 or row >= len(df):
        return {"error": f"Fila {row} fuera de rango (0 - {len(df)-1})."}
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    pred = risk_model.predict([X.iloc[row]])[0]
    prob = risk_model.predict_proba([X.iloc[row]])[0].tolist()
    
    return {
        "row": row,
        "features": X.iloc[row].to_dict(),
        "prediction": int(pred),
        "probabilities": prob
    }
