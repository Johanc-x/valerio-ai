# app/agent/registry.py

from ..calculators.black_scholes import calc_black_scholes_internal
from ..calculators.var_montecarlo import var_montecarlo
from ..calculators.var_simple import calculate_var
from ..calculators.capm import calcular_capm
from ..calculators.markowitz import optimizar_portafolio
from ..calculators.montecarlo import calc_montecarlo
from ..ml.model import risk_model
from ..ml.predict_stock import predict_stock

TOOLS = {
    "calc_black_scholes": calc_black_scholes_internal,   # Black-Scholes
    "calc_var_montecarlo": var_montecarlo,               
    "calc_var_simple": calculate_var,                    # VaR simple
    "calc_capm": calcular_capm,                          # CAPM
    "calc_markowitz": optimizar_portafolio,              # Markowitz
    "calc_montecarlo": calc_montecarlo,                  # Monte Carlo Simulation
    "predict_risk_model": risk_model,                    # Modelo ML de riesgo
    "predict_stock": predict_stock                       # Predicción bursátil
}

