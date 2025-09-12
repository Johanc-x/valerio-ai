import yfinance as yf

# Descargar datos de Apple
ticker = yf.Ticker("AAPL")
data = ticker.history(period="1y")

print(data.head())

# Guardar en CSV
data.to_csv("apple_data.csv")
