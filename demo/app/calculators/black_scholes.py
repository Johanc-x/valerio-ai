# demo/app/calculators/black_scholes.py
import math
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

def _N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes(S, K, r, sigma, T, option="call") -> dict:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("Parámetros inválidos para Black–Scholes.")

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option.lower() in ("call", "c", "llamada"):
        price = S * _N(d1) - K * math.exp(-r * T) * _N(d2)
        delta = _N(d1)
    else:  # put
        price = K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)
        delta = _N(d1) - 1.0

    vega = (S * math.sqrt(T) * (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)) / 100.0

    return {"price": price, "delta": delta, "vega": vega, "d1": d1, "d2": d2}

# --- Wrapper interno para agent.py ---
def calc_black_scholes_internal(S, K, r, sigma, T, option="call", lang="es"):
    result = black_scholes(S, K, r, sigma, T, option)
    price = result['price']

    if lang == "es":
        if option in ["call", "llamada"]:
            explanation = (
                f"El precio estimado de la opción CALL es {price:.2f}. "
                "Según el modelo Black–Scholes, este es el valor justo a pagar hoy "
                "para tener derecho a comprar el activo en el futuro al precio pactado."
            )
        else:
            explanation = (
                f"El precio estimado de la opción PUT es {price:.2f}. "
                "Esto representa el valor justo a pagar hoy para tener derecho a vender "
                "el activo en el futuro al precio pactado."
            )
    else:  # Inglés
        if option == "call":
            explanation = (
                f"The estimated price of the CALL option is {price:.2f}. "
                "According to the Black–Scholes model, this is the fair value today "
                "to acquire the right to buy the asset at the agreed strike price."
            )
        else:
            explanation = (
                f"The estimated price of the PUT option is {price:.2f}. "
                "According to the Black–Scholes model, this is the fair value today "
                "to acquire the right to sell the asset at the agreed strike price."
            )
    return {
        "message": explanation,
        "result": result,
        "lang": lang
    }

# --- Modelo de entrada para JSON ---
class BlackScholesIn(BaseModel):
    S: float
    K: float
    r: float
    sigma: float
    T: float
    option: str = "call"
    lang: str = "es"

# --- Endpoint FastAPI (ahora con JSON body) ---
@router.post("/black-scholes")
def calc_black_scholes_endpoint(body: BlackScholesIn):
    return calc_black_scholes_internal(
        S=body.S,
        K=body.K,
        r=body.r,
        sigma=body.sigma,
        T=body.T,
        option=body.option,
        lang=body.lang
    )
