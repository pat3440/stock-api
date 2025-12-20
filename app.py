from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# ----------------------------
# Moving average implementations
# ----------------------------

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    def _calc(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())
    return series.rolling(window=length, min_periods=length).apply(_calc, raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    n = int(length)
    half = max(1, n // 2)
    sqrt_n = max(1, int(np.sqrt(n)))
    wma_half = wma(series, half)
    wma_full = wma(series, n)
    hull_raw = 2 * wma_half - wma_full
    return wma(hull_raw, sqrt_n)

def dema(series: pd.Series, length: int) -> pd.Series:
    e1 = ema(series, length)
    e2 = ema(e1, length)
    return 2 * e1 - e2

def tema(series: pd.Series, length: int) -> pd.Series:
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    return 3 * e1 - 3 * e2 + e3

def zlema(series: pd.Series, length: int) -> pd.Series:
    n = int(length)
    lag = (n - 1) // 2
    price_adj = 2 * series - series.shift(lag)
    return ema(price_adj, n)

MA_MAP = {
    "sma": sma, "ema": ema, "wma": wma, 
    "hma": hma, "dema": dema, "tema": tema, "zlema": zlema,
}

def moving_average(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    key = ma_type.strip().lower()
    return MA_MAP[key](series.astype(float), int(length))

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/price-data", methods=["GET"])
def price_data():
    ticker = (request.args.get("ticker") or "").strip().upper()
    ma1 = request.args.get("ma1", "50")
    ma2 = request.args.get("ma2", "200")
    ma1_type = (request.args.get("ma1_type") or "sma").strip().lower()
    ma2_type = (request.args.get("ma2_type") or "sma").strip().lower()

    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    try:
        ma1_i, ma2_i = int(ma1), int(ma2)
    except:
        return jsonify({"error": "MA lengths must be integers"}), 400

    # ---- Fetch Data ----
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return jsonify({"error": f"No data for {ticker}"}), 400

    # ---- Handle potential multi-index columns (yfinance quirk) ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ---- Extract Series ----
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # ---- Compute MAs ----
    try:
        ma1_values = moving_average(close, ma1_i, ma1_type)
        ma2_values = moving_average(close, ma2_i, ma2_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # ---- Helper: JSON clean-up (NaN to None) ----
    def clean_list(series, round_digits=2):
        return [
            round(float(v), round_digits) if np.isfinite(v) else None 
            for v in series.to_numpy()
        ]

    return jsonify({
        "ticker": ticker,
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "close": clean_list(close),
        "volume": clean_list(volume, 0), # Volume usually doesn't need decimals
        "ma1": ma1_i,
        "ma1_type": ma1_type,
        "ma1_values": clean_list(ma1_values),
        "ma2": ma2_i,
        "ma2_type": ma2_type,
        "ma2_values": clean_list(ma2_values),
    })

if __name__ == "__main__":
    app.run(debug=True)