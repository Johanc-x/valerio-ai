# demo/app/calculators/var_montecarlo.py
from typing import Optional, List, Dict
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io, base64

router = APIRouter()

# --- Retornos sintéticos por defecto (igual que var_simple) ---
_RNG = np.random.default_rng(seed=123)
DEFAULT_RETURNS = _RNG.normal(0.0, 0.01, size=750).astype(float)

def _ensure_returns(returns):
    if returns is None or len(returns) == 0:
        return DEFAULT_RETURNS
    return np.asarray(returns, dtype=float)

def var_montecarlo(
    returns: Optional[List[float]] = None,
    alpha: float = 0.05,
    horizon: int = 1,
    sims: int = 10_000,
    amount: Optional[float] = None,
) -> Dict:
    rets = _ensure_returns(returns)

    mu = rets.mean()
    sigma = rets.std(ddof=1)

    sims_1d = np.random.normal(mu, sigma, size=sims)
    sims_H = sims_1d * np.sqrt(horizon)

    var = np.quantile(sims_H, alpha)
    es = sims_H[sims_H <= var].mean()

    var_mag = float(abs(var))
    es_mag  = float(abs(es))

    out: Dict = {
        "method": "montecarlo",
        "var_ret": var_mag,
        "es_ret":  es_mag,
        "var_pct": 100.0 * var_mag,
        "es_pct":  100.0 * es_mag,
    }
    if amount is not None:
        out["var_money"] = float(amount) * var_mag
        out["es_money"]  = float(amount) * es_mag

    # --- Gráfico de distribución ---
    fig, ax = plt.subplots()
    ax.hist(sims_H, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    ax.axvline(out["var_ret"], color="red", linestyle="--", label="VaR")
    ax.axvline(out["es_ret"], color="orange", linestyle="--", label="ES")
    ax.set_title("Distribución de pérdidas simuladas (Monte Carlo)")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # --- Mensaje amigable ---
    msg = (
        f"VaR Monte Carlo ({int((1-alpha)*100)}% confianza, {horizon} día/s): "
        f"{out['var_pct']:.2f}% (retorno), "
        f"ES ≈ {out['es_pct']:.2f}%."
    )
    if amount:
        msg += f" Equivale a pérdidas de hasta ${out['var_money']:,.2f}."

    return {
        "message": msg,
        "result": out,
        "graph": img_base64
    }

# --- Pydantic Model ---
class VarMontecarloIn(BaseModel):
    returns: Optional[List[float]] = None
    alpha: float = 0.05
    horizon: int = 1
    sims: int = 10000
    amount: Optional[float] = None

# --- Endpoint ---
@router.post("/var-montecarlo")
def calc_var_montecarlo(body: VarMontecarloIn):
    return var_montecarlo(
        returns=body.returns,
        alpha=body.alpha,
        horizon=body.horizon,
        sims=body.sims,
        amount=body.amount,
    )
