import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Market Scout Pro", page_icon="📈", layout="wide")

st.title("📈 Market Scout – Pro Dashboard")

# -------------------------------
# Function: Check stock with MA
# -------------------------------
def check_stock(ticker, ma_days):
    try:
        data = yf.download(ticker, period="6mo", interval="1d")
        if data.empty:
            return "FAIL", f"No data found for {ticker}", None

        data[f"{ma_days}_MA"] = data["Close"].rolling(ma_days).mean()

        # ✅ FIX: use last non-NaN value
        price = data["Close"].iloc[-1]
        ma_value = data[f"{ma_days}_MA"].dropna().iloc[-1]

        if price > ma_value:
            return "PASS", f"{ticker} is trading above its {ma_days}-day moving average ({price:.2f} > {ma_value:.2f}).", data
        else:
            return "FAIL", f"{ticker} is trading below its {ma_days}-day moving average ({price:.2f} < {ma_value:.2f}).", data

    except Exception as e:
        return "FAIL", f"Error checking {ticker}: {e}", None

# -------------------------------
# Function: Trending stocks
# -------------------------------
def get_trending(tickers):
    try:
        df = yf.download(tickers, period="1mo", interval="1d")["Close"]
        changes = (df.iloc[-1] / df.iloc[-5] - 1) * 100  # % change in last 5 days
        return changes.sort_values(ascending=False).to_frame("5d % Change")
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

# -------------------------------
# Function: Portfolio Analyzer
# -------------------------------
def analyze_portfolio(portfolio):
    results = []
    for stock in portfolio:
        ticker = stock["Ticker"]
        qty = stock["Quantity"]
        cost = stock["Cost"]

        try:
            price = yf.download(ticker, period="5d", interval="1d")["Close"].iloc[-1]
            pl = (price - cost) * qty
            pl_pct = (price - cost) / cost * 100

            # Suggestion rules
            ma200 = yf.download(ticker, period="6mo", interval="1d")["Close"].rolling(200).mean().dropna()
            ma200_val = ma200.iloc[-1] if not ma200.empty else None

            if pl_pct < -15:
                action = "SELL 🚨"
            elif pl_pct > 30:
                action = "SELL (take profit)"
            elif ma200_val and price > ma200_val:
                action = "BUY MORE ✅"
            else:
                action = "HOLD 🤝"

            results.append({
                "Ticker": ticker,
                "Quantity": qty,
                "Cost Basis": cost,
                "Current Price": round(price, 2),
                "P/L ($)": round(pl, 2),
                "P/L (%)": round(pl_pct, 2),
                "Action": action
            })
        except:
            continue

    return pd.DataFrame(results)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Controls")
ma_days = st.sidebar.selectbox("Choose moving average window:", [20, 50, 200], index=1)
universe = st.sidebar.text_input("Enter tickers for Trending (comma separated):", "AAPL, MSFT, TSLA, AMD, NVDA, RGTI")

# -------------------------------
# Stock Checker
# -------------------------------
st.subheader("🔎 Stock Checker – PASS/FAIL")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA):")

if ticker:
    result, reason, data = check_stock(ticker.upper(), ma_days)
    if result == "PASS":
        st.success(f"✅ {result}")
    else:
        st.error(f"❌ {result}")
    st.write(reason)

    if data is not None:
        ma_col = f"{ma_days}_MA"
        chart_data = data[["Close", ma_col]].copy()
        chart_data.columns = ["Price", f"{ma_days}-Day MA"]
        st.line_chart(chart_data)

# -------------------------------
# Trending Section
# -------------------------------
st.subheader("🔥 Top Trending Stocks (5d % Change)")
if universe:
    tickers = [t.strip().upper() for t in universe.split(",")]
    trending = get_trending(tickers)

    if not trending.empty:
        # ✅ Color format: green for gains, red for losses
        def color_format(val):
            color = "green" if val > 0 else "red"
            return f"color: {color}; font-weight: bold;"

        st.dataframe(trending.style.applymap(color_format, subset=["5d % Change"]))
    else:
        st.warning("No trending data available.")

# -------------------------------
# Portfolio Section
# -------------------------------
st.subheader("💼 Portfolio Analyzer")
st.markdown("Enter your positions below (Ticker, Quantity, Cost Basis).")

with st.form("portfolio_form"):
    tickers_input = st.text_area("Enter portfolio (Ticker,Quantity,Cost) one per line:",
                                 "AAPL,10,150\nTSLA,5,700\nRGTI,20,15")
    submitted = st.form_submit_button("Analyze Portfolio")

if submitted:
    portfolio = []
    for line in tickers_input.splitlines():
        try:
            t, q, c = line.split(",")
            portfolio.append({"Ticker": t.strip().upper(), "Quantity": int(q), "Cost": float(c)})
        except:
            continue

    df_portfolio = analyze_portfolio(portfolio)
    if not df_portfolio.empty:
        st.dataframe(df_portfolio)
    else:
        st.warning("No valid portfolio data entered.")
