import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Market Scout Pro Dashboard", layout="wide")

st.title("📊 Market Scout – Pro Dashboard")

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
ma_days = st.sidebar.selectbox("Choose moving average window:", [20, 50, 100, 200], index=1)
tickers_input = st.sidebar.text_input(
    "Enter tickers for Trending (comma separated):",
    "AAPL, MSFT, TSLA, AMD, NVDA, RGTI"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# === Stock Checker Section ===
st.subheader("🔎 Stock Checker – PASS/FAIL")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA):", "AAPL")

def get_data(ticker, ma_days):
    try:
        df = yf.download(ticker, period="6mo")
        df[f"{ma_days}_MA"] = df["Close"].rolling(ma_days).mean()
        return df
    except Exception as e:
        return None

if ticker:
    data = get_data(ticker, ma_days)

    if data is not None and not data.empty:
        close_price = data["Close"].iloc[-1]
        ma_value = data[f"{ma_days}_MA"].dropna().iloc[-1]

        # PASS/FAIL Logic
        if close_price > ma_value:
            st.success(f"✅ PASS\n\n{ticker} is trading above its {ma_days}-day moving average ({close_price:.2f} > {ma_value:.2f}).")
        else:
            st.error(f"❌ FAIL\n\n{ticker} is trading below its {ma_days}-day moving average ({close_price:.2f} < {ma_value:.2f}).")

        # Show Live Price
        live_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        st.metric(label=f"💲 Live Price ({ticker})", value=f"${live_price:.2f}")

        # === Chart with Live Price Overlay ===
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data["Close"], label="Price (Daily Close)", color="blue")
        ax.plot(data.index, data[f"{ma_days}_MA"], label=f"{ma_days}-Day MA", color="orange")

        # Add live price marker
        pct_diff = ((live_price - ma_value) / ma_value) * 100
        ax.axhline(live_price, color="red", linestyle="--", linewidth=1.2)
        ax.scatter(data.index[-1], live_price, color="red", zorder=5)

        # Color % diff text
        color = "green" if pct_diff >= 0 else "red"
        ax.text(data.index[-1], live_price,
                f" Live: ${live_price:.2f} ({pct_diff:+.2f}%)",
                color=color, fontsize=9, verticalalignment="bottom")

        ax.set_title(f"{ticker.upper()} Price vs {ma_days}-Day MA")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

    else:
        st.error(f"Error fetching data for {ticker}")

# === Top Trending Stocks ===
st.subheader("🔥 Top Trending Stocks (5d % Change)")

trending = {}
for t in tickers:
    try:
        df = yf.download(t, period="5d")
        if not df.empty:
            change = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
            trending[t] = change
    except:
        pass

if trending:
    trend_df = pd.DataFrame(trending.items(), columns=["Ticker", "5d % Change"]).sort_values("5d % Change", ascending=False)
    def colorize(val):
        color = "green" if val > 0 else "red"
        return f"color: {color}"
    st.dataframe(trend_df.style.applymap(colorize, subset=["5d % Change"]), height=250)

# === Portfolio Analyzer ===
st.subheader("💼 Portfolio Analyzer")
portfolio_input = st.text_area("Enter your positions below (Ticker, Quantity, Cost Basis) one per line:",
                               "AAPL,10,150\nTSLA,5,700\nAMD,20,100")

portfolio = []
for line in portfolio_input.splitlines():
    try:
        t, q, c = line.split(",")
        portfolio.append((t.strip().upper(), int(q), float(c)))
    except:
        pass

if portfolio:
    results = []
    for t, q, c in portfolio:
        try:
            live_price = yf.Ticker(t).history(period="1d")["Close"].iloc[-1]
            pnl = (live_price - c) * q
            decision = "✅ Hold / Buy More" if live_price > c else "❌ Consider Selling"
            results.append((t, q, c, live_price, pnl, decision))
        except:
            pass

    if results:
        port_df = pd.DataFrame(results, columns=["Ticker", "Quantity", "Cost Basis", "Live Price", "P/L ($)", "Decision"])
        st.dataframe(port_df, height=300)
