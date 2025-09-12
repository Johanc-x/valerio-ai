# demo/app/calculators/markowitz.py
import numpy as np
import matplotlib.pyplot as plt
import io, base64
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# --- Modelo de entrada ---
class MarkowitzIn(BaseModel):
    rendimientos: list[float]
    covarianzas: list[list[float]]
    rf: float = 0.02

def optimizar_portafolio(rendimientos: list, covarianzas: list, rf: float = 0.02) -> dict:
    rendimientos = np.array(rendimientos)
    covarianzas = np.array(covarianzas)
    n = len(rendimientos)

    n_portafolios = 5000
    resultados = np.zeros((3, n_portafolios))
    pesos_array = []

    for i in range(n_portafolios):
        pesos = np.random.random(n)
        pesos /= np.sum(pesos)
        retorno = np.dot(pesos, rendimientos)
        riesgo = np.sqrt(np.dot(pesos.T, np.dot(covarianzas, pesos)))
        sharpe = (retorno - rf) / riesgo

        resultados[0, i] = riesgo
        resultados[1, i] = retorno
        resultados[2, i] = sharpe
        pesos_array.append(pesos)

    max_sharpe_idx = np.argmax(resultados[2])
    mejor_riesgo, mejor_retorno, mejor_sharpe = resultados[:, max_sharpe_idx]
    mejores_pesos = pesos_array[max_sharpe_idx]

    # --- Gráfico ---
    fig, ax = plt.subplots()
    scatter = ax.scatter(resultados[0,:], resultados[1,:], c=resultados[2,:], cmap='viridis')
    ax.scatter(mejor_riesgo, mejor_retorno, color='red', marker='*', s=200, label="Portafolio Óptimo")
    ax.set_xlabel("Riesgo (σ)")
    ax.set_ylabel("Retorno esperado")
    ax.legend()
    plt.colorbar(scatter, label="Sharpe Ratio")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    message = (
        f"Según Markowitz, el portafolio óptimo asigna los pesos {np.round(mejores_pesos, 2)}. "
        f"Retorno esperado: {mejor_retorno:.2%}, Riesgo: {mejor_riesgo:.2%}, Sharpe: {mejor_sharpe:.2f}."
    )

    return {
        "message": message,
        "result": {
            "weights": np.round(mejores_pesos, 2).tolist(),
            "retorno": float(mejor_retorno),
            "riesgo": float(mejor_riesgo),
            "sharpe": float(mejor_sharpe),
        },
        "graph": img_base64
    }

# --- Endpoint ---
@router.post("/markowitz")
def markowitz_endpoint(body: MarkowitzIn):
    return optimizar_portafolio(body.rendimientos, body.covarianzas, body.rf)
