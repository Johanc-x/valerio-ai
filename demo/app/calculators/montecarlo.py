import numpy as np
import matplotlib.pyplot as plt
import io, base64
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class MonteCarloIn(BaseModel):
    S0: float
    mu: float
    sigma: float
    T: float
    steps: int
    sims: int

def calc_montecarlo(S0: float, mu: float, sigma: float, T: float, steps: int, sims: int):
    dt = T / steps
    prices = np.zeros((steps + 1, sims))
    prices[0] = S0

    for t in range(1, steps + 1):
        rand = np.random.normal(0, 1, sims)
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)

    final_prices = prices[-1]
    expected_price = np.mean(final_prices)
    volatility = np.std(final_prices)

    # 📊 Gráfico (20 trayectorias)
    fig, ax = plt.subplots()
    for i in range(min(20, prices.shape[1])):
        ax.plot(np.linspace(0, T, steps+1), prices[:, i], alpha=0.5)

    ax.set_title("Simulación Monte Carlo")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Precio")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return {
        "message": (
            f"Monte Carlo completado con {sims} simulaciones. "
            f"Precio esperado: {expected_price:.2f}, "
            f"Volatilidad: {volatility:.4f}"
        ),
        "result": {
            "expected_price": float(expected_price),
            "volatility": float(volatility),
            "simulations": sims
        },
        "graph": img_base64
    }

# --- Endpoint FastAPI ---
@router.post("/montecarlo")
def montecarlo_endpoint(body: MonteCarloIn):
    return calc_montecarlo(**body.dict())
