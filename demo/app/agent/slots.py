import re, ast

def _to_float(x: str) -> float:
    return float(x.replace(",", ".").strip())

def parse_percent(val: str):
    v = float(val.replace("%", ""))
    return v / 100.0 if "%" in val else v

def parse_money(val: str):
    v = val.lower().replace(",", "").replace("$", "").strip()
    mult = 1.0
    if v.endswith("k"):
        mult, v = 1e3, v[:-1]
    if v.endswith("m"):
        mult, v = 1e6, v[:-1]
    return float(v) * mult


# ---------------- Black-Scholes ----------------
def extract_bs(text: str) -> dict:
    out = {}
    num = r"([-+]?\d+(?:[\.,]\d+)?)"

    S = re.search(r"\bS\s*[:=]?\s*" + num, text, re.I)
    K = re.search(r"\bK\s*[:=]?\s*" + num, text, re.I)
    r_ = re.search(r"\br\s*[:=]?\s*" + num + r"\s*%?", text, re.I)
    sg = re.search(r"\b(?:sigma|vol)\s*[:=]?\s*" + num + r"\s*%?", text, re.I)
    T = re.search(r"\bT\s*[:=]?\s*" + num, text, re.I)
    opt = re.search(r"\b(call|put|llamada|venta)\b", text, re.I)

    if S: out["S"] = _to_float(S.group(1))
    if K: out["K"] = _to_float(K.group(1))
    if r_:
        rv = _to_float(r_.group(1))
        out["r"] = rv / 100 if "%" in r_.group(0) or rv > 1 else rv
    if sg:
        sv = _to_float(sg.group(1))
        out["sigma"] = sv / 100 if "%" in sg.group(0) or sv > 1 else sv
    if T: out["T"] = _to_float(T.group(1))
    if opt:
        out["option"] = "call" if opt.group(1).lower().startswith(("call", "llam")) else "put"
    return out

# ---------------- Helpers ----------------
def _to_float(s: str) -> float | None:
    """Convierte string a float de forma robusta."""
    try:
        s = s.replace(",", ".").strip()
        if s.endswith("."):
            s = s[:-1]
        return float(s)
    except ValueError:
        return None

# ---------------- VaR (simple, histórico, montecarlo, ewma) ----------------
def extract_var(text: str) -> dict:
    out = {}

    # método: hist | ewma | montecarlo
    m = re.search(r"\b(hist[oó]rico|historical|ewma|monte\s*carlo|montecarlo)\b", text, re.I)
    if m:
        method = m.group(1).lower().replace(" ", "")
        if "hist" in method:
            out["method"] = "historic"
        elif "ewma" in method:
            out["method"] = "ewma"
        elif "monte" in method:
            out["method"] = "montecarlo"

    # lambda (para ewma)
    lam = re.search(r"(?:lambda|λ)\s*[:=]?\s*([\d\.,]+)", text, re.I)
    if lam:
        val = _to_float(lam.group(1))
        if val is not None:
            out["lambda"] = val

    # horizonte
    hz = re.search(r"(?:horizonte|hor|horizon)\s*[:=]?\s*(\d+)", text, re.I) or \
         re.search(r"\b(\d+)\s*(?:d[ií]as|days|d)\b", text, re.I) or \
         re.search(r"over\s+(\d+)\s+days", text, re.I)
    if hz:
        out["horizon"] = int(hz.group(1))

    # monto
    amt = re.search(r"(?:monto|amount)\s*[:=]?\s*([\d\.,]+)\s*[kKmM]?", text, re.I)
    if amt:
        val = _to_float(amt.group(1))
        if val is not None:
            raw = amt.group(0).lower()
            if "k" in raw:
                val *= 1000
            elif "m" in raw:
                val *= 1_000_000
            out["amount"] = val

    # nivel/alpha
    lvl = re.search(r"(?:nivel|alfa|alpha|at|level)\s*[:=]?\s*([\d\.,]+)\s*%?", text, re.I) or \
          re.search(r"\b(\d{1,3})\s*%?\b", text, re.I)
    if lvl:
        x = _to_float(lvl.group(1))
        if x is not None:
            if x > 1:   # 95 → 0.05
                out["alpha"] = round(1 - x / 100.0, 6)
            else:       # 0.05 → 0.05
                out["alpha"] = round(x, 6)

    return out

# ---------------- Monte Carlo ----------------
import re

def extract_montecarlo(text: str) -> dict:
    slots = {}
    text = text.lower()

    patterns = {
        "s0": r"s0\s*=\s*([\d\.]+)",
        "mu": r"mu\s*=\s*([\d\.]+)",
        "sigma": r"sigma\s*=\s*([\d\.]+)",
        "t": r"t\s*=\s*([\d\.]+)",
        "steps": r"(?:steps|pasos)\s*=\s*(\d+)",
        "sims": r"(?:sims|simulaciones)\s*=\s*(\d+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)  # 👈 ignora mayúsculas/minúsculas
        if m:
            try:
                slots[key] = float(m.group(1)) if key not in ("steps", "sims") else int(m.group(1))
            except:
                pass

    return slots

# ---------------- ML row ----------------
def extract_row(text: str):
    # Busca expresiones como: "row=7", "fila 7", "predice la fila 10"
    m = re.search(r"(?:row|fila)\s*[:=]?\s*(\d+)", text, re.I)
    if m:
        return {"row": int(m.group(1))}
    return {}

# ---------------- CAPM ----------------
def extract_capm(text: str) -> dict:
    out = {}

    # rf puede venir como "0.02", "2%", "rf=2", etc.
    rf = re.search(r"rf\s*=?\s*([\d\.,]+)\s*%?", text, re.I)
    beta = re.search(r"beta\s*=?\s*([\d\.,]+)", text, re.I)
    rm = re.search(r"rm\s*=?\s*([\d\.,]+)\s*%?", text, re.I)

    if rf:
        val = float(rf.group(1).replace(",", "."))
        out["rf"] = val / 100 if "%" in rf.group(0) or val > 1 else val

    if beta:
        out["beta"] = float(beta.group(1).replace(",", "."))

    if rm:
        val = float(rm.group(1).replace(",", "."))
        out["rm"] = val / 100 if "%" in rm.group(0) or val > 1 else val

    return out

# ---------------- Markowitz ----------------
def extract_markowitz(text: str) -> dict:
    out = {}
    rend = re.search(r"(?:rend|returns)[=:\s]*([\[\]0-9.,\s-]+)", text, re.I)
    cov = re.search(r"(?:cov|covariance)[=:\s]*([\[\]0-9.,\s\-\[\]]+)", text, re.I)

    if rend:
        try:
            out["rendimientos"] = ast.literal_eval(rend.group(1))
        except Exception as e:
            out["error"] = f"Formato inválido en rendimientos: {e}"
    if cov:
        try:
            out["covarianzas"] = ast.literal_eval(cov.group(1))
        except Exception as e:
            out["error"] = f"Formato inválido en covarianzas: {e}"
        return out
    
# ---------------- Stock Prediction ----------------
def extract_stock_predict(text: str) -> dict:
    out = {}
    clean = re.sub(r"[^A-Z0-9ÁÉÍÓÚÜÑ ]", " ", text.upper())

    # --- Captura ticker ---
    VALID_TICKERS = {"AAPL", "TSLA", "MSFT", "AMZN"}
    words = re.findall(r"\b[A-Z]{2,5}\b", text.upper())
    for w in words:
        if w in VALID_TICKERS:
            out["ticker"] = w
            break

    # --- Captura días (soporta español e inglés, con o sin "próximos") ---
    d = re.search(r"(\d+)\s*(DAY|DAYS?|DIA|DIAS?)", clean)
    out["days"] = int(d.group(1)) if d else 1

    # --- Captura modelo (alias en ES/EN, con tilde y sin tilde) ---
    if re.search(r"XGBOOST", clean):
        out["model"] = "xgboost_reg"
    elif re.search(r"SVM|SVR|VECTOR", clean):
        out["model"] = "svr"
    elif re.search(r"FOREST|BOSQUE", clean):
        out["model"] = "random_forest_reg"
    elif re.search(r"LINEAR|REGRESION|REGRESSION", clean):
        out["model"] = "linear_regression"
    else:
        out["model"] = "xgboost_reg"  # default seguro
    return out
