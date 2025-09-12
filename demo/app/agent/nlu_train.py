from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

# app/agent/nlu_train.py

# dataset y modelo
DATA = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PKL = MODEL_DIR / "nlu_intents.pkl"

def train():
    df = pd.read_csv(DATA)
    X = df["text"].astype(str)
    y = df["intent"].astype(str)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",           # char n-grams: robusto con pocas frases
            ngram_range=(3, 5),
            sublinear_tf=True,
            lowercase=True,
            min_df=1,
            max_df=0.95
        )),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ])

    # Elegir n_splits según el tamaño mínimo de clase para evitar errores
    class_counts = y.value_counts()
    n_splits = max(2, min(5, int(class_counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    print(f"CV f1_macro: {f1.mean():.3f} +/- {f1.std():.3f}  ({n_splits} folds)")

    # Entrena en TODO el dataset y guarda el modelo final
    pipe.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PKL)
    print("Saved ->", MODEL_PKL)


if __name__ == "__main__":
    train()
