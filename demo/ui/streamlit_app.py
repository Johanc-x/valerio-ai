import streamlit as st
import requests

st.set_page_config(page_title="Valerio AI Demo", layout="wide")

st.sidebar.title("Valerio AI")
api_base = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")

tab_pred, tab_bs, tab_var = st.tabs(["🔮 Predicción (ML)", "📈 Black-Scholes", "📉 VaR Monte Carlo"])

with tab_pred:
    st.header("Predicción de Riesgo (modelo real)")
    row_idx = st.number_input("row_idx", min_value=0, value=0, step=1)
    if st.button("Predecir", type="primary"):
        try:
            url = f"{api_base}/ml/predict"
            r = requests.get(url, params={"row_idx": row_idx}, timeout=20)
            r.raise_for_status()
            data = r.json()
            st.json(data)
            if "prob" in data and data["prob"] is not None:
                st.metric("Prob(riesgo=1)", f"{data['prob']:.3f}")
            if "pred" in data:
                st.metric("Predicción", str(data["pred"]))
            if "real" in data and data["real"] is not None:
                st.metric("Real (si existe)", str(data["real"]))
        except Exception as e:
            st.error(str(e))

with tab_bs:
    st.header("Black-Scholes (europeo)")
    S = st.number_input("S (spot)", value=100.0)
    K = st.number_input("K (strike)", value=100.0)
    r = st.number_input("r (tasa anual)", value=0.02, format="%.4f")
    sigma = st.number_input("sigma (vol anual)", value=0.20, format="%.4f")
    T = st.number_input("T (años)", value=0.5)
    option = st.selectbox("Tipo", ["call", "put"])
    if st.button("Calcular BS"):
        try:
            url = f"{api_base}/calc/black_scholes"
            r = requests.get(url, params=dict(S=S, K=K, r=r, sigma=sigma, T=T, option=option), timeout=10)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

with tab_var:
    st.header("VaR Monte Carlo")
    price = st.number_input("Precio inicial", value=100.0)
    mu = st.number_input("mu (media anual)", value=0.0, format="%.4f")
    sigma = st.number_input("sigma (vol anual)", value=0.20, format="%.4f")
    days = st.number_input("Horizonte (días)", value=1, step=1)
    cl = st.slider("Confianza", min_value=0.80, max_value=0.99, value=0.95)
    sims = st.number_input("Simulaciones", value=10000, step=1000)
    if st.button("Simular VaR"):
        try:
            url = f"{api_base}/calc/var"
            r = requests.get(url, params=dict(price=price, mu=mu, sigma=sigma, days=days, cl=cl, sims=sims), timeout=20)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(str(e))
