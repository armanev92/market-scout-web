import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Market Scout GPT", layout="wide")
st.title("📈 Market Scout – PASS/FAIL Stock Checker")

def check_stock(ticker: str):
    try:
        data = yf.download(ticker, period="6mo")
        if data.empty:
            return "❌ FAIL", f"Could not fetch data for {ticker}."

        # Calculate 50-day moving average
        data["50_MA"] = data["Close"].rolling(window=50).mean()

        # Get the most recent closing price & MA as floats
        latest_price = float(data["Close"].iloc[-1])
        latest_ma = float(data["50_MA"].iloc[-1])

        if pd.isna(latest_ma):
            return "❌ FAIL", f"Not enough data yet to calculate 50-day MA for {ticker}."

        if latest_price > latest_ma:
            return "✅ PASS", f"{ticker} is trading above its 50-day moving average ({latest_price:.2f} > {latest_ma:.2f})."
        else:
            return "❌ FAIL", f"{ticker} is trading below its 50-day moving average ({latest_price:.2f} < {latest_ma:.2f})."

    except Exception as e:
        return "❌ FAIL", f"Error checking {ticker}: {e}"

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA):")

if ticker:
    result, reason = check_stock(ticker.upper())
    st.subheader(res
