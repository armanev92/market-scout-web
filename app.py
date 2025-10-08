import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Market Scout GPT", layout="wide")
st.title("📈 Market Scout – PASS/FAIL Stock Checker")

def check_stock(ticker: str):
    try:
        data = yf.download(ticker, period="6mo")  # last 6 months
        if data.empty:
            return "❌ FAIL", f"Could not fetch data for {ticker}."

        data["50_MA"] = data["Close"].rolling(window=50).mean()
        latest_price = data["Close"].iloc[-1]
        latest_ma = data["50_MA"].iloc[-1]

        if latest_price > latest_ma:
            return "✅ PASS", f"{ticker} is trading above its 50-day moving average ({latest_price:.2f} > {latest_ma:.2f})."
        else:
            return "❌ FAIL", f"{ticker} is trading below its 50-day moving average ({latest_price:.2f} < {latest_ma:.2f})."
    except Exception as e:
        return "❌ FAIL", f"Error checking {ticker}: {e}"

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA):")

if ticker:
    result, reason = check_stock(ticker.upper())
    st.subheader(result)
    st.write(reason)
