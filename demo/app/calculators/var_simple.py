# demo/app/calculators/var_simple.py
from __future__ import annotations
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
import io, base64
import matplotlib.pyplot as plt

router = APIRouter()

# --- Dataset sintético por defecto ---
_RNG = np.random.default_rng(seed=42)
DEFAULT_RETURNS = _RNG.normal(0.0, 0.01, size=750).astype(float)

def _ensure_returns(returns):
    if returns is None or len(returns) == 0:
        return DEFAULT_RETURNS
    return np.asarray(returns, dtype=float)

# --- Pydantic Model ---
class VarSimpleIn(BaseModel):
    returns: list[float] | None = None
    confidence: float = 0.95

# --- Endpoint con gráfico ---
@router.post("/var")
def calculate_var(body: VarSimpleIn):
    """
    Calcula el VaR simple a partir de retornos explícitos y genera un gráfico.
    """
    returns = _ensure_returns(body.returns)
    confidence = body.confidence

    # cálculo VaR (percentil de cola)
    var = np.percentile(returns, (1 - confidence) * 100)
    var_mag = abs(float(var))
    var_pct = var_mag * 100

    # --- Gráfico ---
    fig, ax = plt.subplots()
    ax.hist(returns, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
    ax.axvline(var, color="red", linestyle="--", linewidth=2,
               label=f"VaR {confidence*100:.0f}%")
    ax.set_title("Distribución de Retornos y VaR")
    ax.set_xlabel("Retorno")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # ✅ Mensaje claro y homogéneo
    msg = (
        f"VaR Simple ({int(confidence*100)}% confianza): "
        f"{var_pct:.2f}% (retorno). "
        "Esto significa que, bajo condiciones normales, "
        "las pérdidas no deberían superar este nivel."
    )

    return {
        "message": msg,
        "result": {
            "var_ret": var_mag,
            "var_pct": var_pct
        },
        "graph": img_base64
    }
