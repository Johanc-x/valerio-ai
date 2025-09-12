import os
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from app.agent.registry import TOOLS
import re

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# limpiar espacios/saltos de línea invisibles
if api_key:
    api_key = api_key.strip()

client = OpenAI(api_key=api_key)

router = APIRouter()

# --- Diccionarios auxiliares ---
SYMBOL_MAP = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "meta": "META"
}

MODEL_MAP = {
    "random forest": "random_forest_reg",
    "xgboost": "xgboost_reg",
    "linear": "linear_regression",
    "svr": "svr"
}

class Query(BaseModel):
    question: str

@router.post("/ask")
async def ask_valerio(query: Query):
    user_text = query.question.strip()
    lower_text = user_text.lower()
    graph = None
    context = ""

    # Detección simple de idioma
    lang = "es" if any(c in "áéíóúñ¿¡" for c in user_text) or " el " in lower_text else "en"

    try:
        # --- 1. Modelos ML ---
        company = next((name for name in SYMBOL_MAP if name in lower_text), None)
        if company:
            chosen_model = next((v for k, v in MODEL_MAP.items() if k in lower_text), "random_forest_reg")

            # Detectar número de días en la pregunta (default=5)
            match = re.search(r"(\d+)\s*(day|days|dia|dias|día|días)", lower_text)
            days_requested = int(match.group(1)) if match else 5

            result = TOOLS["predict_stock"](SYMBOL_MAP[company], days=days_requested, model=chosen_model)
            graph = result.get("graph")
            if lang == "es":
                context = (
                    f"Predicciones para {company.title()} ({SYMBOL_MAP[company]}). "
                    f"Modelo: {chosen_model}. Próximos {result['days']} días, "
                    f"muestras de predicción: {result['predictions'][:3]}."
                )
            else:
                context = (
                    f"Predictions for {company.title()} ({SYMBOL_MAP[company]}). "
                    f"Model: {chosen_model}. Next {result['days']} days, "
                    f"sample predictions: {result['predictions'][:3]}."
                )

        # --- 2. Black-Scholes ---
        elif "black scholes" in lower_text:
            result = TOOLS["calc_black_scholes"](S=150, K=145, T=1, r=0.05, sigma=0.2, option="call")
            graph = result.get("graph")
            context = result["message"]

        # --- 3. Markowitz ---
        elif "markowitz" in lower_text:
            result = TOOLS["calc_markowitz"](
                [0.1, 0.15, 0.2],
                [[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]]
            )
            graph = result.get("graph")
            context = result["message"]

        # --- 4. Monte Carlo Simulation ---
        elif "monte carlo" in lower_text:
            result = TOOLS["calc_montecarlo"](S0=100, mu=0.05, sigma=0.2, T=1.0, steps=252, sims=10000)
            graph = result.get("graph")
            context = result["message"]

        # --- 5. CAPM ---
        elif "capm" in lower_text:
            result = TOOLS["calc_capm"](rf=0.02, beta=1.1, rm=0.08)
            graph = result.get("graph")
            context = result["message"]

        # --- 6. VaR Monte Carlo ---
        elif "var" in lower_text and "monte carlo" in lower_text:
            result = TOOLS["calc_var"](alpha=0.05, horizon=5, sims=10000, amount=200000)
            graph = result.get("graph")
            context = result["message"]

        # --- 7. VaR simple ---
        elif "var" in lower_text:
            result = TOOLS["calc_var_simple"](returns=[-0.02, 0.01, 0.015, -0.01], confidence=0.95)
            graph = result.get("graph")
            context = result["message"]

                # --- Easter egg / Demo reel ---
        elif "ready to make an impact in london & berlin" in lower_text or "ready to make an impact in london and berlin" in lower_text:
            return {
                "answer": "Absolutely captain, I'm with you on this mission.",
                "graph": None
            }
        
        else:
            context = user_text  # fallback: usar texto directo

    except Exception as e:
        context = f"⚠️ Error running calculation: {str(e)}"

    # --- OpenAI paso final ---
    system_msg = (
        "You are Valerio AI, a financial intelligence system. "
        "Always respond as a professional analyst, based ONLY on the provided context. "
        f"Respond strictly in {'Spanish' if lang == 'es' else 'English'}."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": context},
        ],
        temperature=0.6,
        max_tokens=150,
    )

    return {
        "answer": response.choices[0].message.content,
        "graph": graph,
    }
