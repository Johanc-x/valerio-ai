from pathlib import Path
import joblib

HERE = Path(__file__).resolve().parent
MODEL_PKL = HERE / "models" / "nlu_intents.pkl"

_model = None
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PKL)
    return _model

def predict_intent(text: str) -> str:
    mdl = load_model()
    return str(mdl.predict([text])[0])
