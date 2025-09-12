from .nlu import predict_intent
from .slots import extract_bs, extract_var, extract_row, extract_capm, extract_markowitz
from .registry import TOOLS
from langdetect import detect
from .slots import extract_stock_predict, extract_montecarlo
from app.calculators.black_scholes import calc_black_scholes_internal

def _fmt_money(x: float) -> str:
    return f"{x:,.2f}"

def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _normalize_question(text: str) -> str:
    """
    Limpia saludos y palabras de cortesía para que el NLU detecte mejor la intención.
    """
    import re
    patrones = [
        r"\bhola\b", r"\bhey\b", r"\bvalerio\b", r"\bpor favor\b",
        r"\bcalcula(r|me)?\b", r"\bdime\b", r"\bpuedes\b"
    ]
    out = text.lower()
    for pat in patrones:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def answer(q: str) -> dict:
    q_clean = _normalize_question(q)   # limpiamos la pregunta
    intent = predict_intent(q_clean)
    lang = "es"
    try:
        lang = detect(q)
    except:
        pass

    resp = {"intent": intent, "question": q}

    HELP_MSG = {
        "es": (
            "¡Hola! Soy Valerio.\n"
            "• Black-Scholes: 'black scholes S=100 K=105 r=2.5% sigma=25% T=0.5'\n"
            "• ML: 'predict row 7'\n"
            "• VaR: 'VaR 95% 5 días 200k [histórico|ewma]'\n"
            "• CAPM: 'CAPM rf=0.02 beta=1.1 rm=0.08'\n"
            "• Markowitz: 'Markowitz rend=[0.1,0.15,0.2] cov=[[...]]'\n"
            "¿Qué te gustaría probar?"
        ),
        "en": (
            "Hi! I’m Valerio.\n"
            "• Black-Scholes: 'black scholes S=100 K=105 r=2.5% sigma=25% T=0.5'\n"
            "• ML: 'predict row 7'\n"
            "• VaR: 'VaR 95% 5 days 200k [historical|ewma]'\n"
            "• CAPM: 'CAPM rf=0.02 beta=1.1 rm=0.08'\n"
            "• Markowitz: 'Markowitz returns=[0.1,0.15,0.2] cov=[[...]]'\n"
            "What would you like to try?"
        )
    }

    try:

        # ---------- Black-Scholes ----------
        if intent == "calc_black_scholes" or "black scholes" in q_clean.lower():
            try:
                slots = extract_bs(q_clean)
                need = [k for k in ("S", "K", "r", "sigma", "T") if k not in slots]
                if need:
                    msg = (
                        f"Faltan parámetros: {need} (usa S= K= r= sigma= T=)." if lang == "es"
                        else f"Missing parameters: {need} (use S= K= r= sigma= T=)."
                    )
                    return {**resp, "need": need, "message": msg, "result": None, "lang": lang}

                data = calc_black_scholes_internal(
                    S=slots["S"],
                    K=slots["K"],
                    r=slots["r"],
                    sigma=slots["sigma"],
                    T=slots["T"],
                    option=slots.get("option", "call"),
                    lang=lang
                )
                return {**resp, **data}

            except Exception as e:
                msg = "Error en el cálculo de Black-Scholes" if lang == "es" else "Error in Black-Scholes calculation"
                return {**resp, "error": str(e), "message": msg, "result": None, "lang": lang}

        # ---------- VaR ----------
        if intent in ("calc_var", "calc_var_simple") or "var" in q_clean.lower():
            slots = extract_var(q_clean)
            faltan = [k for k in ("alpha", "horizon") if k not in slots]
            if faltan:
                msg = (
                    "Especifica nivel (ej. VaR 95%) y horizonte (ej. 5 días)." if lang == "es"
                    else "Specify level (e.g. VaR 95%) and horizon (e.g. 5 days)."
                )
                return {**resp, "need": faltan, "message": msg}

            try:
                if intent == "calc_var":
                    res = TOOLS["calc_var"](
                        alpha=slots["alpha"],
                        horizon=slots["horizon"],
                        amount=slots.get("amount"),
                        method=slots.get("method", "montecarlo"),
                    )
                else:
                    res = TOOLS["calc_var_simple"](
                        alpha=slots["alpha"],
                        horizon=slots["horizon"],
                        amount=slots.get("amount"),
                        method=slots.get("method", "ewma"),
                        lam=slots.get("lambda", 0.94),
                    )
            except Exception as e:
                msg = "Error en cálculo VaR" if lang == "es" else "Error in VaR calculation"
                return {**resp, "error": str(e), "message": msg}

            nivel = int(round((1 - slots["alpha"]) * 100))
            if "var_money" in res:
                msg = (
                    f"VaR {res['method'].upper()} {nivel}% a {slots['horizon']} día(s): {_fmt_money(res['var_money'])}"
                    if lang == "es" else
                    f"VaR {res['method'].upper()} {nivel}% over {slots['horizon']} day(s): {_fmt_money(res['var_money'])}"
                )
            else:
                msg = (
                    f"VaR {res['method'].upper()} {nivel}% a {slots['horizon']} día(s): {_fmt_pct(res['var_ret'])} (retorno)"
                    if lang == "es" else
                    f"VaR {res['method'].upper()} {nivel}% over {slots['horizon']} day(s): {_fmt_pct(res['var_ret'])} (return)"
                )
            return {**resp, "result": res, "message": msg}

        # ---------- Monte Carlo ----------
        if intent == "calc_montecarlo" or "monte carlo" in q_clean.lower():
            slots = extract_montecarlo(q_clean)
            need = [k for k in ("s0", "mu", "sigma", "t", "steps", "sims") if k not in slots]
            if need:
                msg = (
                    f"Faltan parámetros: {need}. Ejemplo: 'monte carlo S0=100 mu=0.05 sigma=0.2 T=1 steps=252 sims=10000'" if lang == "es"
                    else f"Missing parameters: {need}. Example: 'monte carlo S0=100 mu=0.05 sigma=0.2 T=1 steps=252 sims=10000'"
                )
                return {**resp, "need": need, "message": msg}

            try:
                resultado = TOOLS["calc_montecarlo"](
                    S0=slots["s0"],
                    mu=slots["mu"],
                    sigma=slots["sigma"],
                    T=slots["t"],
                    steps=slots["steps"],
                    sims=slots["sims"]
                )
                msg = (
                    f"Monte Carlo finalizado con {slots['sims']} simulaciones. "
                    f"Precio esperado: {resultado['expected_price']:.2f}, "
                    f"Riesgo (volatilidad): {resultado['volatility']:.4f}"
                    if lang == "es" else
                    f"Monte Carlo completed with {slots['sims']} simulations. "
                    f"Expected price: {resultado['expected_price']:.2f}, "
                    f"Risk (volatility): {resultado['volatility']:.4f}"
                )
                return {**resp, "result": resultado, "message": msg}

            except Exception as e:
                msg = "Error en simulación Monte Carlo" if lang == "es" else "Error in Monte Carlo simulation"
                return {**resp, "error": str(e), "message": msg}

        # ---------- CAPM ----------
        if intent == "calc_capm" or "capm" in q_clean.lower():
            slots = extract_capm(q_clean)
            faltan = [k for k in ("rf", "beta", "rm") if k not in slots]
            if faltan:
                msg = (
                    "Faltan parámetros. Ejemplo: 'CAPM rf=0.02 beta=1.1 rm=0.08'" if lang == "es"
                    else "Missing parameters. Example: 'CAPM rf=0.02 beta=1.1 rm=0.08'"
                )
                return {**resp, "need": faltan, "message": msg}

            resultado = TOOLS["calc_capm"](rf=slots["rf"], beta=slots["beta"], rm=slots["rm"])
            msg = (
                f"CAPM calculado con rf={slots['rf']}, beta={slots['beta']}, rm={slots['rm']} → Retorno esperado: {resultado:.2%}"
                if lang == "es" else
                f"CAPM calculated with rf={slots['rf']}, beta={slots['beta']}, rm={slots['rm']} → Expected return: {resultado:.2%}"
            )
            return {**resp, "result": {"expected_return": resultado}, "message": msg}

        # ---------- Markowitz ----------
        if intent == "calc_markowitz" or "markowitz" in q_clean.lower():
            slots = extract_markowitz(q_clean)
            if "rendimientos" not in slots or "covarianzas" not in slots:
                msg = (
                    "Debes indicar rendimientos y covarianzas. Ejemplo: 'Markowitz rend=[0.1,0.15,0.2] cov=[[...]]'" if lang == "es"
                    else "You must specify returns and covariances. Example: 'Markowitz returns=[0.1,0.15,0.2] cov=[[...]]'"
                )
                return {**resp, "need": ["rendimientos", "covarianzas"], "message": msg}

            try:
                resultado = TOOLS["calc_markowitz"](slots["rendimientos"], slots["covarianzas"])
                msg = (
                    f"Según Markowitz, el portafolio óptimo asigna los pesos {resultado['weights']}. "
                    f"Retorno esperado: {resultado['retorno']:.2%}, Riesgo: {resultado['riesgo']:.2%}, Sharpe: {resultado['sharpe']:.2f}."
                    if lang == "es" else
                    f"According to Markowitz, the optimal portfolio assigns the weights {resultado['weights']}. "
                    f"Expected return: {resultado['retorno']:.2%}, Risk: {resultado['riesgo']:.2%}, Sharpe: {resultado['sharpe']:.2f}."
                )
                return {**resp, "result": resultado, "message": msg}
            except Exception as e:
                msg = (
                    f"Error en el cálculo de Markowitz: {str(e)}" if lang == "es"
                    else f"Error in Markowitz calculation: {str(e)}"
                )
                return {**resp, "error": str(e), "message": msg}
                
        # ---------- Predict Risk ----------
        if intent == "predict_risk":
            import re, numpy as np

            def get_number(key, text):
                m = re.search(rf"{key}\s*=?\s*([0-9]*\.?[0-9]+)", q_clean)
                return float(m.group(1)) if m else None

            zscore = get_number("zscore", q_clean)
            volatility = get_number("volatility", q_clean)
            returns = get_number("returns", q_clean)
            debt_ratio = get_number("debt_ratio", q_clean)

            faltan = [k for k, v in {
                "zscore": zscore,
                "volatility": volatility,
                "returns": returns,
                "debt_ratio": debt_ratio
            }.items() if v is None]

            if faltan:
                msg = (
                    f"Faltan parámetros: {faltan}. Ejemplo: 'predice riesgo zscore=2.1 volatility=0.15 returns=0.08 debt_ratio=0.3'"
                    if lang == "es" else
                    f"Missing parameters: {faltan}. Example: 'predict risk zscore=2.1 volatility=0.15 returns=0.08 debt_ratio=0.3'"
                )
                return {**resp, "need": faltan, "message": msg}

            X = np.array([[zscore, volatility, returns, debt_ratio]])
            pred = TOOLS["predict_risk_model"].predict(X)[0]
            prob = TOOLS["predict_risk_model"].predict_proba(X)[0].tolist()

            label = "BAJO" if pred == 0 else "ALTO"
            label_en = "LOW" if pred == 0 else "HIGH"
            prob_percent = round(max(prob) * 100, 2)

            msg = (
                f"Riesgo {label} con {prob_percent}% de probabilidad." if lang == "es"
                else f"Risk {label_en} with {prob_percent}% probability."
            )
            return {**resp, "result": {"label": label, "prob": prob_percent}, "message": msg}

        # ---------- Stock Prediction ----------
        if intent == "predict_stock":
            slots = extract_stock_predict(q_clean)

            if "ticker" not in slots:
                msg = (
                    "Indica un ticker (ej. AAPL, TSLA)." if lang == "es"
                    else "Specify a ticker (e.g., AAPL, TSLA)."
                )
                return {**resp, "need": ["ticker"], "message": msg}

            ticker = slots["ticker"]
            days = slots.get("days", 1)
            # Mapeo de alias de modelos
            model_aliases = {
                "xgboost": "xgboost_reg",
                "xgboost_reg": "xgboost_reg",
                "linear": "linear_regression",
                "linear_regression": "linear_regression",
                "random_forest": "random_forest_reg",
                "random_forest_reg": "random_forest_reg",
                "svr": "svr",
                "support_vector": "svr",
            }

            raw_model = slots.get("model", "xgboost_reg")
            model = model_aliases.get(raw_model, raw_model)

            try:
                res = TOOLS["predict_stock"](ticker=ticker, days=days, model=model)

                # Mensaje breve SOLO encabezado (sin repetir los valores)
                msg = (
                    f"Predicción de precios para {ticker} usando {model} (horizonte {days} días)."
                    if lang == "es" else
                    f"Price prediction for {ticker} using {model} (horizon {days} days)."
                )

                return {**resp, "result": res, "message": msg}

            except Exception as e:
                msg = (
                    f"Error en la predicción bursátil: {str(e)}" if lang == "es"
                    else f"Error in stock prediction: {str(e)}"
                )
                return {**resp, "result": {}, "error": str(e), "message": msg}

        # ---------- Help ----------
        if intent in ("help", "other", None):
            return {**resp, "message": HELP_MSG["es" if lang == "es" else "en"]}

    except Exception as e:
        return {**resp, "error": str(e)}
