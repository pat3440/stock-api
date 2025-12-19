from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd


import os
print("=== RENDER DEBUG ===")
print("CWD:", os.getcwd())
print("FILES:", os.listdir("."))
print("====================")

app = Flask(__name__)


# For now, allow requests from anywhere. After it's working, we can restrict this to your WordPress domain.
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Temporary: helps confirm what routes Render thinks exist.
# Remove this once everything is working.
@app.route("/debug/routes", methods=["GET"])
def debug_routes():
    return jsonify(sorted([r.rule for r in app.url_map.iter_rules()]))

@app.route("/api/price-data", methods=["GET"])
def price_data():
    ticker = (request.args.get("ticker") or "").strip().upper()
    ma1 = request.args.get("ma1", "50")
    ma2 = request.args.get("ma2", "200")

    # Validate inputs
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    try:
        ma1_i = int(ma1)
        ma2_i = int(ma2)
        if ma1_i <= 0 or ma2_i <= 0:
            raise ValueError()
    except Exception:
        return jsonify({"error": "ma1 and ma2 must be positive integers"}), 400

    # Fetch 5y daily data
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return jsonify({"error": f"No data returned for {ticker}"}), 400

    close = df["Close"].astype("float")
    ma1_s = close.rolling(ma1_i).mean()
    ma2_s = close.rolling(ma2_i).mean()

    dates = df.index.strftime("%Y-%m-%d").tolist()

    def to_list(series: pd.Series):
        # Convert NaN -> None for JSON; otherwise round floats
        out = []
        for v in series.tolist():
            if v != v:  # NaN check
                out.append(None)
            else:
                out.append(round(float(v), 2))
        return out

    return jsonify({
        "ticker": ticker,
        "ma1": ma1_i,
        "ma2": ma2_i,
        "dates": dates,
        "close": to_list(close),
        "ma1_values": to_list(ma1_s),
        "ma2_values": to_list(ma2_s),
    })
