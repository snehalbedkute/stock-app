from flask import Flask, request, render_template, jsonify   # ✅ added jsonify
from flask_cors import CORS
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
import numpy as np
import requests  # For live news fetching

# ===== ML imports for LSTM =====
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__, template_folder="../templates")
CORS(app)

# ✅ HOME route (works now)
@app.route('/')
def home():
    return render_template("index.html")   # index.html must be inside "templates" folder


# ✅ AUTOFETCH route for stock suggestions
@app.route("/get-stocks", methods=["GET"])
def get_stocks():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    try:
        search = yf.Ticker(query)
        info = search.info

        results = []
        if "symbol" in info and "shortName" in info:
            results.append({
                "symbol": info["symbol"],
                "name": info["shortName"]
            })

        return jsonify(results)

    except Exception:
        return jsonify([])


def create_line_graph(dates, values, title):
    """Returns a <img> tag with a base64 PNG of a simple line graph."""
    plt.figure(figsize=(10, 5))
    plt.plot(dates, values, marker='o', linestyle='-', color='green')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_url = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f"<img src='data:image/png;base64,{graph_url}'/>"


# ------------------------------
# PAST: UNCHANGED
# ------------------------------
@app.route('/past-data', methods=['POST'])
def past_data():
    data = request.get_json()
    symbol = data.get('symbol')
    start = data.get('start_date')
    end = data.get('end_date')

    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            return "<p>No data found for this range.</p>"

        df = df[['Open', 'Close', 'High', 'Low']]

        # Check if range >= 1 year
        start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
        days_diff = (end_date - start_date).days

        if days_diff >= 365:
            # Summarize into monthly data
            df_monthly = df.resample('M').mean()
            df_monthly.index = df_monthly.index.strftime('%b %Y')

            html_table = df_monthly.to_html()
            graph_html = create_line_graph(
                df_monthly.index, df_monthly['Close'], f'{symbol.upper()} Monthly Avg Closing Prices'
            )

            return f"<h3>📊 1-Year Summary for {symbol.upper()}</h3>{html_table}<br><h3>📈 Monthly Trend</h3>{graph_html}"
        else:
            # Keep your existing detailed daily mode
            html_table = df.to_html()
            graph_html = create_line_graph(
                df.index.strftime('%Y-%m-%d'), df['Close'], f'{symbol.upper()} Closing Prices'
            )
            return f"<h3>📊 Stock Data for {symbol.upper()}</h3>{html_table}<br><h3>📈 Price Trend</h3>{graph_html}"

    except Exception as e:
        return f"<p>Error: {str(e)}</p>"


# ------------------------------
# FUTURE: LSTM MODEL + LIVE NEWS + SENTIMENT
# ------------------------------
def lstm_predict_next_7(symbol, today):
    start_date = today - datetime.timedelta(days=400)
    df = yf.download(symbol, start=start_date, end=today)

    if df.empty or len(df) < 120:
        raise Exception("Not enough history to train an LSTM model (need at least ~120 trading days).")

    close = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(close)

    lookback = 60
    X, y = [], []
    for i in range(lookback, len(close_scaled)):
        X.append(close_scaled[i - lookback:i, 0])
        y.append(close_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_window = close_scaled[-lookback:].reshape(1, lookback, 1)

    preds_scaled = []
    for _ in range(7):
        next_scaled = model.predict(last_window, verbose=0)[0][0]
        preds_scaled.append(next_scaled)
        new_window = np.append(last_window[0, 1:, 0], next_scaled)
        last_window = new_window.reshape(1, lookback, 1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    preds = [float(np.round(p, 2)) for p in preds]

    dates = [(today + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    return dates, preds


@app.route('/future-data', methods=['POST'])
def future_data():
    data = request.get_json()
    symbol = data.get('symbol')

    try:
        today = datetime.date.today()
        dates, predictions = lstm_predict_next_7(symbol, today)

        prediction_table = "<table border='1'><tr><th>Date</th><th>Predicted Price</th></tr>"
        for d, p in zip(dates, predictions):
            prediction_table += f"<tr><td>{d}</td><td>{p}</td></tr>"
        prediction_table += "</table>"

        graph_html = create_line_graph(dates, predictions, f"{symbol.upper()} Future Price Prediction (LSTM)")

        api_key = "1e48ce7ba1454174a1e1648d226b8764"
        news_url = (
            "https://newsapi.org/v2/everything"
            f"?q={symbol}&sortBy=publishedAt&language=en&apiKey={api_key}"
        )

        headlines = []
        try:
            news_resp = requests.get(news_url, timeout=8)
            news_json = news_resp.json()
            if news_json.get("articles"):
                for article in news_json["articles"][:3]:
                    title = article.get("title") or ""
                    source = (article.get("source") or {}).get("name") or ""
                    headlines.append(f"{title} — {source}")
            if not headlines:
                headlines = [f"No recent news found for {symbol}"]
        except Exception:
            headlines = [f"News fetch failed; proceeding without news."]

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg = sum(scores) / len(scores) if scores else 0.0

        suggestion = "Buy" if avg > 0.2 else ("Sell" if avg < -0.2 else "Hold")

        sentiment_html = "<div style='background: rgba(0,0,0,0.6); padding: 15px; border-radius: 8px; text-align: left; display: inline-block; max-width: 900px;'>"
        for h, s in zip(headlines, scores):
            sentiment_html += (
                f"<p style='margin-bottom:8px;'>"
                f"<strong style='color:white;'>{h}</strong><br>"
                f"<span style='color:lightgray;'>Score: {round(s, 4)}</span></p>"
            )
        sentiment_html += "</div>"

        return f"""
            <h3>🔮 Future Price Prediction for {symbol.upper()} (LSTM)</h3>
            {prediction_table}
            <br><h3>📈 Predicted Trend</h3>
            {graph_html}
            <h3>📰 Recent News + Sentiment</h3>
            {sentiment_html}
            <h3>✅ Recommendation: {suggestion}</h3>
        """

    except Exception as e:
        return f"<p>Error: {str(e)}</p>"
    
@app.route('/past')
def past_page():
    return render_template("past.html")

@app.route('/future')
def future_page():
    return render_template("future.html")


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
