# ml/train_stock_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib

# Ruta del dataset
DATA = Path(__file__).resolve().parents[1] / "data" / "apple_data.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PKL = MODEL_DIR / "xgboost_apple.pkl"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features a partir de precios de Apple."""
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Volatility5"] = df["Close"].pct_change().rolling(window=5).std()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(window=5).mean()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)  # 1 si sube mañana
    return df.dropna()

def train():
    df = pd.read_csv(DATA)
    df = build_features(df)

    X = df[["Return", "MA5", "MA10", "Volatility5", "Volume_Ratio"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("=== Stock Prediction Model (Apple) ===")
    print("Accuracy:", round(acc, 3))
    print("F1 Score:", round(f1, 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PKL)
    print("✅ Modelo guardado en:", MODEL_PKL)

if __name__ == "__main__":
    train()
