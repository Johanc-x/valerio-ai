# demo/app/calculators/capm.py
import matplotlib.pyplot as plt
import io, base64
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

def calcular_capm(rf: float, beta: float, rm: float) -> dict:
    expected_return = rf + beta * (rm - rf)

    # --- Gráfico ---
    fig, ax = plt.subplots()
    ax.axhline(rf, color="red", linestyle="--", label="Rf (Libre de riesgo)")
    ax.plot([0, beta], [rf, expected_return], marker="o", label="Línea CAPM")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Retorno esperado")
    ax.set_title("Capital Asset Pricing Model (CAPM)")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    message = (
        f"Según el modelo CAPM, el activo debería rendir aproximadamente "
        f"{expected_return:.2%}, considerando un retorno de mercado del {rm:.2%}, "
        f"una tasa libre de riesgo del {rf:.2%} y una beta de {beta:.2f}."
    )

    return {
        "message": message,
        "result": {"expected_return": expected_return},
        "graph": img_base64
    }

# --- Modelo de entrada ---
class CapmIn(BaseModel):
    rf: float
    beta: float
    rm: float

# --- Endpoint FastAPI ---
@router.post("/capm")
def calc_capm(body: CapmIn):
    return calcular_capm(body.rf, body.beta, body.rm)
