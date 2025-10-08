import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Market Scout GPT", layout="wide")
st.title("📈 Market Scout – PASS/FAIL Stock Checker")

# --- Inputs ---
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA):")
ma_days = st.selectbox("Choose moving average window:", [20, 50, 200], index=1)

def check_stock(ticker: str, ma_days: int):
    try:
        data = yf.download(ticker, period="6mo")
        if data.empty:
            return "❌ FAIL", f"Could not fetch data for {ticker}.", None

        data[f"{ma_days}_MA"] = data["Close"].rolling(window=ma_days).mean()

        latest_price = float(data["Close"].iloc[-1])
        latest_ma = float(data[f"{ma_days}_MA"].iloc[-1])

        if pd.isna(latest_ma):
            return "❌ FAIL", f"Not enough data yet to calculate {ma_days}-day MA for {ticker}.", data

        if latest_price > latest_ma:
            return "✅ PASS", f"{ticker} is trading above its {ma_days}-day moving average ({latest_price:.2f} > {latest_ma:.2f}).", data
        else:
            return "❌ FAIL", f"{ticker} is trading below its {ma_days}-day moving average ({latest_price:.2f} < {latest_ma:.2f}).", data

    except Exception as e:
        return "❌ FAIL", f"Error checking {ticker}: {e}", None

if ticker:
    result, reason, data = check_stock(ticker.upper(), ma_days)
    st.subheader(result)
    st.write(reason)

    if data is not None:
        st.line_chart(data[["Close", f"{ma_days}_MA"]])
