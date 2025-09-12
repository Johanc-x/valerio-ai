# demo/app/ml/train_models.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import io, base64

# ============================
# 1. Cargar dataset
# ============================
data_path = Path(__file__).resolve().parent.parent / "data" / "apple_data.csv"
df = pd.read_csv(data_path)

# Features
df["Return"] = df["Close"].pct_change()
df["MA5"] = df["Close"].rolling(5).mean()
df["MA10"] = df["Close"].rolling(10).mean()
df["Volatility5"] = df["Return"].rolling(5).std()
df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()
df.dropna(inplace=True)

# Target = próximo precio
df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

X = df[["Return", "MA5", "MA10", "Volatility5", "Volume_Ratio"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ============================
# 2. Definir modelos regresores
# ============================
models = {
    "xgboost_reg": XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
    "linear_regression": LinearRegression(),
    "random_forest_reg": RandomForestRegressor(n_estimators=200, random_state=42),
    "svr": SVR(kernel="rbf")
}

results = []
models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# ============================
# 3. Entrenar y evaluar
# ============================
for name, model in models.items():
    print(f"\n=== Entrenando {name.upper()} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MSE: {mse:.3f}, R2: {r2:.3f}")

    # Guardar modelo
    save_path = models_dir / f"{name}_apple.pkl"
    joblib.dump(model, save_path)
    print(f"✅ Modelo guardado en: {save_path}")

    # Guardar métricas
    results.append({
        "model": name,
        "mse": mse,
        "r2": r2
    })

# ============================
# 4. Resumen comparativo
# ============================
print("\n=== Comparación de Modelos ===")
for r in results:
    print(f"{r['model']:20s} | MSE: {r['mse']:.3f} | R2: {r['r2']:.3f}")
