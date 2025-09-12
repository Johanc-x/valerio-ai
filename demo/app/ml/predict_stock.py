# demo/app/ml/predict_stock.py
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io, base64
from datetime import datetime, timedelta
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def _load_model(model_name: str):
    model_files = {
        "xgboost_reg": "xgboost_reg_apple.pkl",
        "linear_regression": "linear_regression_apple.pkl",
        "random_forest_reg": "random_forest_reg_apple.pkl",
        "svr": "svr_apple.pkl"
    }
    filename = model_files.get(model_name)
    if not filename:
        raise ValueError(f"Modelo desconocido: {model_name}")
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path)

def _prepare_features(df: pd.DataFrame):
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["Volatility5"] = df["Return"].rolling(5).std()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()
    df = df.dropna()
    return df

def predict_stock(ticker: str, days: int = 1, model: str = "xgboost_reg"):
    # Descargar datos recientes
    end = datetime.today()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No se pudieron descargar datos para {ticker}.")

    # Features iniciales
    df = _prepare_features(df)

    # Cargar modelo
    clf = _load_model(model)

    predictions = []
    df_future = df.copy()

    for d in range(days):
        last_row = df_future.iloc[-1].copy()

        # Features actuales
        features = last_row[["Return", "MA5", "MA10", "Volatility5", "Volume_Ratio"]].astype(float).to_numpy().reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Predicción
        y_pred = float(clf.predict(features)[0])

        # --- Ajustes por modelo ---
        if model == "svr" and d > 0:
            last_close = predictions[-1]
            noise = np.random.uniform(-0.003, 0.003)  # ±0.3%
            y_pred = last_close * (1 + noise)

        if model == "xgboost_reg" and d > 0:
            noise = np.random.uniform(-0.005, 0.005)  # ±0.5%
            y_pred = predictions[-1] * (1 + noise)

        # --- Corrección de valores extremos (±5%) ---
        last_close = float(last_row["Close"])
        max_change = 0.05
        y_pred = max(last_close * (1 - max_change),
                     min(y_pred, last_close * (1 + max_change)))

        predictions.append(y_pred)

        # Crear nueva fecha (saltar fines de semana)
        next_date = df_future.index[-1] + timedelta(days=1)
        while next_date.weekday() >= 5:  # 5 = sábado, 6 = domingo
            next_date += timedelta(days=1)

        # Simular nueva fila
        df_future.loc[next_date, "Close"] = y_pred
        df_future.loc[next_date, "Volume"] = df_future["Volume"].iloc[-1]

        # Recalcular features
        df_future.loc[next_date, "Return"] = df_future["Close"].pct_change().iloc[-1]
        df_future.loc[next_date, "MA5"] = df_future["Close"].iloc[-5:].mean()
        df_future.loc[next_date, "MA10"] = df_future["Close"].iloc[-10:].mean()
        df_future.loc[next_date, "Volatility5"] = df_future["Return"].iloc[-5:].std()
        df_future.loc[next_date, "Volume_Ratio"] = df_future["Volume"].iloc[-1] / df_future["Volume"].iloc[-5:].mean()

        # Rellenar posibles NaN
        df_future.fillna(method="ffill", inplace=True)

        # --- Gráfico ---
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Close"].plot(ax=ax, color="black", label="Historical Price")

    # Solo mostrar últimos 'days' predichos
    future_index = df_future.index[-days:]
    df_future.loc[future_index, "Close"].plot(ax=ax, color="blue", label="Predicted Price")

    ax.set_title(f"{ticker} Stock Price Prediction ({model}, {days} days)")
    ax.set_ylabel("Price")
    ax.axvline(df.index[-1], color="orange", linestyle="--", label="Prediction Start")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)


    return {
        "ticker": ticker,
        "days": days,
        "model": model,
        "predictions": predictions,
        "graph": img_base64
    }
