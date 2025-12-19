from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)

# Allow requests from anywhere
# After everything is stable, we can restrict this to your WordPress domain.
CORS(app)

# ----------------------------
# Moving average implementations
# ----------------------------

def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length, min_periods=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average (standard trading convention)."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def wma(series: pd.Series, length: int) -> pd.Series:
    """
    Weighted Moving Average (linear weights 1..length; newest has highest weight).
    """
    weights = np.arange(1, length + 1, dtype=float)

    def _calc(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window=length, min_periods=length).apply(_calc, raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    """
    Hull Moving Average (Alan Hull):
      HMA(n) = WMA( 2*WMA(price, n/2) - WMA(price, n), sqrt(n) )
    """
    n = int(length)
    if n < 1:
        raise ValueError("HMA length must be >= 1")

    half = max(1, n // 2)
    sqrt_n = max(1, int(np.sqrt(n)))

    wma_half = wma(series, half)
    wma_full = wma(series, n)
    hull_raw = 2 * wma_half - wma_full
    return wma(hull_raw, sqrt_n)

def dema(series: pd.Series, length: int) -> pd.Series:
    """
    Double Exponential Moving Average:
      DEMA = 2*EMA(price, n) - EMA(EMA(price, n), n)
    """
    e1 = ema(series, length)
    e2 = ema(e1, length)
    return 2 * e1 - e2

def tema(series: pd.Series, length: int) -> pd.Series:
    """
    Triple Exponential Moving Average:
      TEMA = 3*EMA1 - 3*EMA2 + EMA3
    """
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    return 3 * e1 - 3 * e2 + e3

def zlema(series: pd.Series, length: int) -> pd.Series:
    """
    Zero-Lag Exponential Moving Average (ZLEMA):
      lag = floor((n - 1) / 2)
      price_adj = 2*price - price.shift(lag)
      ZLEMA = EMA(price_adj, n)
    """
    n = int(length)
    if n < 1:
        raise ValueError("ZLEMA length must be >= 1")

    lag = (n - 1) // 2
    price_adj = 2 * series - series.shift(lag)
    return ema(price_adj, n)

MA_MAP = {
    "sma": sma,
    "ema": ema,
    "wma": wma,
    "hma": hma,
    "dema": dema,
    "tema": tema,
    "zlema": zlema,
}

def moving_average(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    """Dispatcher to compute MA by name."""
    if not isinstance(ma_type, str):
        raise ValueError("ma_type must be a string")

    key = ma_type.strip().lower()
    if key not in MA_MAP:
        raise ValueError(
            f"Unsupported ma_type '{ma_type}'. Supported: {', '.join(sorted(MA_MAP.keys()))}"
        )

    n = int(length)
    if n <= 0:
        raise ValueError("length must be a positive integer")

    return MA_MAP[key](series.astype(float), n)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/price-data", methods=["GET"])
def price_data():
    ticker = (request.args.get("ticker") or "").strip().upper()

    # lengths
    ma1 = request.args.get("ma1", "50")
    ma2 = request.args.get("ma2", "200")

    # types (NEW)
    ma1_type = (request.args.get("ma1_type") or "sma").strip().lower()
    ma2_type = (request.args.get("ma2_type") or "sma").strip().lower()

    # ---- Validate inputs ----
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    try:
        ma1_i = int(ma1)
        ma2_i = int(ma2)
        if ma1_i <= 0 or ma2_i <= 0:
            raise ValueError()
    except Exception:
        return jsonify({"error": "ma1 and ma2 must be positive integers"}), 400

    if ma1_type not in MA_MAP:
        return jsonify({
            "error": f"Unsupported ma1_type '{ma1_type}'",
            "supported": sorted(MA_MAP.keys())
        }), 400

    if ma2_type not in MA_MAP:
        return jsonify({
            "error": f"Unsupported ma2_type '{ma2_type}'",
            "supported": sorted(MA_MAP.keys())
        }), 400

    # ---- Fetch 5-year daily data ----
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return jsonify({"error": f"No data returned for {ticker}"}), 400

    # ---- Force Close into a 1-D Series ----
    close_obj = df["Close"]
    close = close_obj.iloc[:, 0] if isinstance(close_obj, pd.DataFrame) else close_obj
    close = close.astype("float")

    # ---- Moving averages (NEW: selectable type) ----
    try:
        ma1_s = moving_average(close, ma1_i, ma1_type)
        ma2_s = moving_average(close, ma2_i, ma2_type)
    except Exception as e:
        return jsonify({"error": "Failed to compute moving averages", "details": str(e)}), 400

    # ---- Dates ----
    dates = df.index.strftime("%Y-%m-%d").tolist()

    # ---- Helper: convert Series → JSON-safe list ----
    def to_list(values):
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]

        out = []
        for v in values.to_numpy().tolist():
            if v != v:  # NaN check
                out.append(None)
            else:
                out.append(round(float(v), 2))
        return out

    # ---- Response ----
    return jsonify({
        "ticker": ticker,
        "ma1": ma1_i,
        "ma1_type": ma1_type,
        "ma2": ma2_i,
        "ma2_type": ma2_type,
        "supported_ma_types": sorted(MA_MAP.keys()),
        "dates": dates,
        "close": to_list(close),
        "ma1_values": to_list(ma1_s),
        "ma2_values": to_list(ma2_s),
    })
