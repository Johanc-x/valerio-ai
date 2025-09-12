# demo/app/main.py
from typing import List, Optional
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import io, base64
from fastapi.responses import JSONResponse
from app import routes_openai

from .schemas import AskIn, AskOut
from .agent.agent import answer as agent_answer
from .ml.valerio_core_adapter import predict_by_row_index  

# Routers de calculadoras
# main.py
from .calculators.black_scholes import router as black_scholes_router
from .calculators.var_simple import router as var_simple_router
from .calculators.var_montecarlo import router as var_montecarlo_router
from .calculators.capm import router as capm_router
from .calculators.markowitz import router as markowitz_router
from .calculators.montecarlo import router as montecarlo_router


app = FastAPI(title="Valerio AI - MVP", version="0.1.0")

app.include_router(black_scholes_router, prefix="/calc", tags=["Black-Scholes"])
app.include_router(var_simple_router, prefix="/calc", tags=["VaR"])
app.include_router(var_montecarlo_router, prefix="/calc", tags=["VaR Montecarlo"])
app.include_router(capm_router, prefix="/calc", tags=["CAPM"])
app.include_router(markowitz_router, prefix="/calc", tags=["Markowitz"])
app.include_router(routes_openai.router, prefix="/valerio", tags=["Valerio AI"])
app.include_router(montecarlo_router, prefix="/calc", tags=["Monte Carlo"])

# --- Cargar modelo de riesgo ---
MODEL_PATH = Path(__file__).resolve().parent / "ml" / "models" / "risk_xgboost.pkl"
risk_model = joblib.load(MODEL_PATH)
features = ["zscore", "volatility", "returns", "debt_ratio"]
print("Clases del modelo:", risk_model.classes_)

# CORS básico para poder llamar desde el frontend (puedes limitar orígenes luego)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# --- Endpoints “oficiales” que consumirá el frontend ---
@app.get("/ml/predict")
def ml_predict(row: int = Query(..., ge=0)):
    # usando la función pública del adapter
    return predict_by_row_index(row)

# --- Endpoint de predicción de riesgo con gráfico ---
@app.post("/predict_risk")
def predict_risk(zscore: float, volatility: float, returns: float, debt_ratio: float):
    X = np.array([[zscore, volatility, returns, debt_ratio]])
    pred = risk_model.predict(X)[0]
    prob = risk_model.predict_proba(X)[0].tolist()

    label = "BAJO" if pred == 0 else "ALTO"
    prob_percent = round(max(prob) * 100, 2)

    responses = [
        f"Según mis cálculos, el riesgo de esta empresa es {label}, con una probabilidad del {prob_percent}%.",
        f"Tras analizar los indicadores financieros, estimo un riesgo {label} (confianza {prob_percent}%).",
        f"Los resultados sugieren que la compañía presenta un nivel de riesgo {label}, con {prob_percent}% de certeza.",
        f"Con base en el Z-Score, volatilidad y deuda, el diagnóstico es: riesgo {label}, con probabilidad del {prob_percent}%."
    ]
    response = random.choice(responses)

    if label == "ALTO":
        response += " Se recomienda revisar el ratio de deuda y la volatilidad, que parecen elevados."
    else:
        response += " Los indicadores sugieren una posición financiera relativamente estable."

    # --- Gráfico de barras: Probabilidades ---
    fig, ax = plt.subplots()
    ax.bar(["BAJO", "ALTO"], prob, color=["green", "red"])
    ax.set_title("Probabilidad de Riesgo")
    ax.set_ylabel("Probabilidad")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return {
        "message": response,
        "graph": img_base64
    }

# --- Endpoint de gráfico de importancia de features ---
@app.get("/risk_feature_importance")
def risk_feature_importance():
    importance = risk_model.feature_importances_

    plt.figure(figsize=(6,4))
    plt.bar(features, importance, color="steelblue")
    plt.title("Importancia de variables en el modelo de riesgo")
    plt.ylabel("Peso")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return {"image_base64": img_base64}

