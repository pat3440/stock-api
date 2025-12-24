from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ----------------------------
# Moving average implementations (Unchanged)
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
    if key not in MA_MAP:
        raise ValueError(f"Invalid MA type: {ma_type}")
    return MA_MAP[key](series.astype(float), int(length))

@app.route("/api/price-data", methods=["GET"])
def price_data():
    ticker = (request.args.get("ticker") or "").strip().upper()
    ma1 = request.args.get("ma1", "50")
    ma2 = request.args.get("ma2", "200")
    ma1_type = (request.args.get("ma1_type") or "sma").strip().lower()
    ma2_type = (request.args.get("ma2_type") or "sma").strip().lower()
    
    # NEW: Get start_date from frontend
    start_date_str = request.args.get("start_date", "2020-01-01")

    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    try:
        ma1_i, ma2_i = int(ma1), int(ma2)
        requested_start = pd.to_datetime(start_date_str)
    except:
        return jsonify({"error": "Invalid parameters"}), 400

    # ---- Buffer Logic ----
    # We fetch extra data before the start_date so MAs have time to "warm up"
    # A 2x buffer of the longest MA is usually safe for HMA/TEMA
    buffer_days = max(ma1_i, ma2_i) * 2
    fetch_start = requested_start - timedelta(days=buffer_days)

    # ---- Fetch Data ----
    df = yf.download(ticker, start=fetch_start.strftime('%Y-%m-%d'), interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return jsonify({"error": f"No data for {ticker}"}), 400

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].astype(float)

    # ---- Compute MAs ----
    try:
        ma1_values = moving_average(close, ma1_i, ma1_type)
        ma2_values = moving_average(close, ma2_i, ma2_type)
        ma_diff = ma1_values - ma2_values
        
        # Combine into a single DataFrame for easy date filtering
        results_df = pd.DataFrame({
            "close": close,
            "ma1_values": ma1_values,
            "ma2_values": ma2_values,
            "ma_diff": ma_diff
        }, index=df.index)

        # Filter out the buffer: only return data from the requested start date onwards
        final_df = results_df[results_df.index >= requested_start]
        
    except Exception as e:
        return jsonify({"error": f"Calculation Error: {str(e)}"}), 400

    # ---- Helper: JSON clean-up ----
    def clean_list(series, round_digits=2):
        return [
            round(float(v), round_digits) if np.isfinite(v) else None 
            for v in series.to_numpy()
        ]

    return jsonify({
        "ticker": ticker,
        "dates": final_df.index.strftime("%Y-%m-%d").tolist(),
        "close": clean_list(final_df["close"]),
        "ma1_values": clean_list(final_df["ma1_values"]),
        "ma2_values": clean_list(final_df["ma2_values"]),
        "ma_diff": clean_list(final_df["ma_diff"])
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)