import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Market Scout Pro", page_icon="📈", layout="wide")
st.title("📈 Market Scout – Pro Dashboard")

# -------------------------------
# Helpers
# -------------------------------
def _to_float(x) -> float:
    """Safely convert Series/array/scalar to float."""
    if isinstance(x, (pd.Series, pd.Index, list, tuple, np.ndarray)):
        return float(pd.Series(x).dropna().iloc[-1])
    return float(x)

def _live_price_now(ticker: str) -> float | None:
    """Try fast_info, then fallback to 1m history."""
    try:
        fi = yf.Ticker(ticker).fast_info
        last = fi.get("last_price", None)
        if last is not None and not np.isnan(last):
            return float(last)
    except Exception:
        pass
    try:
        h = yf.Ticker(ticker).history(period="1d", interval="1m")["Close"].dropna()
        if not h.empty:
            return float(h.iloc[-1])
    except Exception:
        pass
    return None

# -------------------------------
# Function: Intraday chart + AI trading signal
# -------------------------------
def intraday_chart_with_signal(ticker: str):
    try:
        df = yf.download(ticker, period="1d", interval="5m", auto_adjust=True, progress=False)
        if df.empty:
            return None, "No intraday data available."

        # Indicators
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        latest = df.iloc[-1]
        price = float(latest["Close"])
        ema9 = float(latest["EMA9"])
        ema21 = float(latest["EMA21"])
        rsi = float(latest["RSI"])

        # AI signal
        if price > ema9 > ema21 and rsi < 70:
            signal = "BUY ✅ (bullish momentum)"
        elif price < ema21 or rsi > 70:
            signal = "SELL ❌ (weakening or overbought)"
        else:
            signal = "HOLD 🤝 (sideways)"

        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Candles"
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"],
                                 line=dict(color="blue", width=1), name="EMA 9"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"],
                                 line=dict(color="orange", width=1), name="EMA 21"))

        fig.update_layout(title=f"{ticker} Intraday (5m) Candlestick",
                          xaxis_rangeslider_visible=False,
                          template="plotly_dark",
                          height=400)

        return fig, signal

    except Exception as e:
        return None, f"Error: {e}"

# -------------------------------
# Function: Trending stocks
# -------------------------------
def get_trending(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            close = yf.download(t, period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(name=t)

            if close.shape[0] < 5:
                rows.append({"Ticker": t, "Live Price": None, "5d % Change": None})
                continue

            change_5d = float((close.iloc[-1] / close.iloc[-5] - 1.0) * 100.0)
            live_price = _live_price_now(t)

            rows.append({
                "Ticker": t,
                "Live Price": round(live_price, 2) if live_price else None,
                "5d % Change": round(change_5d, 2)
            })
        except Exception:
            rows.append({"Ticker": t, "Live Price": None, "5d % Change": None})

    return pd.DataFrame(rows)

# -------------------------------
# Function: Portfolio Analyzer
# -------------------------------
def analyze_portfolio(portfolio_rows: list[dict]) -> pd.DataFrame:
    results = []
    for row in portfolio_rows:
        t = str(row["Ticker"]).upper().strip()
        qty = float(row["Quantity"])
        cost = float(row["Cost"])

        try:
            lp = _live_price_now(t)
            if lp is None:
                lp = _to_float(yf.Ticker(t).history(period="5d", interval="1d")["Close"].iloc[-1])

            pnl = (lp - cost) * qty
            pnl_pct = (lp - cost) / cost if cost else np.nan

            hist = yf.download(t, period="6mo", interval="1d", auto_adjust=True, progress=False)
            if not hist.empty:
                ma200_ser = hist["Close"].rolling(200).mean().dropna()
                ma200 = _to_float(ma200_ser.iloc[-1]) if not ma200_ser.empty else np.nan
            else:
                ma200 = np.nan

            if not np.isnan(pnl_pct) and pnl_pct < -0.15:
                action = "SELL 🚨"
                why = "Unrealized loss greater than 15%."
            elif not np.isnan(pnl_pct) and pnl_pct > 0.30:
                action = "SELL (take profit)"
                why = "Unrealized gain greater than 30%."
            elif not np.isnan(ma200) and lp > ma200:
                action = "BUY MORE ✅"
                why = "Price above 200-day MA (long-term uptrend)."
            else:
                action = "HOLD 🤝"
                why = "Within normal range or trend unclear."

            results.append({
                "Ticker": t,
                "Quantity": qty,
                "Cost Basis": round(cost, 2),
                "Live Price": round(lp, 2),
                "P/L ($)": round(pnl, 2),
                "P/L (%)": f"{pnl_pct*100:,.2f}%" if pnl_pct == pnl_pct else "N/A",
                "Action": action,
                "Reason": why
            })
        except Exception:
            continue

    return pd.DataFrame(results)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Controls")
universe = st.sidebar.text_input("Enter tickers for Trending (comma separated):",
                                 "AAPL, MSFT, TSLA, AMD, NVDA, RGTI")

# 🔄 Auto-refresh every 60 sec
st_autorefresh(interval=60 * 1000, key="refresh")

# -------------------------------
# Stock Checker
# -------------------------------
st.subheader("🔎 Stock Checker – AI Intraday Signal")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA):")

if ticker:
    fig, signal = intraday_chart_with_signal(ticker.upper())
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"🤖 AI Trading Signal: {signal}")
    else:
        st.warning(signal)

# -------------------------------
# Trending Section
# -------------------------------
st.subheader("🔥 Top Trending Stocks (5d % Change + Live Price)")
if universe.strip():
    tickers_list = [t.strip().upper() for t in universe.split(",") if t.strip()]
    trending = get_trending(tickers_list)

    if not trending.empty:
        def color_format(val):
            try:
                return f"color: {'green' if val > 0 else 'red'}; font-weight: bold;"
            except Exception:
                return ""
        st.dataframe(trending.style.applymap(color_format, subset=["5d % Change"]), hide_index=True)
    else:
        st.warning("No trending data available.")

# -------------------------------
# Portfolio Section
# -------------------------------
st.subheader("💼 Portfolio Analyzer")
st.markdown("Enter your positions below (Ticker, Quantity, Cost Basis). Example: `AAPL,10,150`")

with st.form("portfolio_form"):
    tickers_input = st.text_area("Portfolio (Ticker,Quantity,Cost)",
                                 "AAPL,10,150\nTSLA,5,700\nRGTI,20,15")
    submitted = st.form_submit_button("Analyze Portfolio")

if submitted:
    portfolio_rows = []
    for line in tickers_input.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        t, q, c = parts
        try:
            portfolio_rows.append({"Ticker": t.upper(), "Quantity": float(q), "Cost": float(c)})
        except Exception:
            continue

    df_portfolio = analyze_portfolio(portfolio_rows)
    if not df_portfolio.empty:
        st.dataframe(df_portfolio, hide_index=True)
        try:
            total_value = float(np.nansum(df_portfolio["Live Price"] * df_portfolio["Quantity"]))
            total_pl = float(np.nansum(df_portfolio["P/L ($)"]))
            c1, c2 = st.columns(2)
            c1.metric("Portfolio Value", f"${total_value:,.2f}")
            c2.metric("Total P/L", f"${total_pl:,.2f}")
        except Exception:
            pass
    else:
        st.warning("No valid portfolio rows parsed. Use lines like `AAPL,10,150`.")

# Footer
st.caption("⚠️ Educational purposes only. This is not investment advice.")
