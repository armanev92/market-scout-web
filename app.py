import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import requests
from textblob import TextBlob

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
    """Crypto (-USD) -> force 1m candles. Stocks -> fast_info fallback."""
    try:
        if "-USD" in ticker:  # crypto handling
            h = yf.Ticker(ticker).history(period="1d", interval="1m")["Close"].dropna()
            if not h.empty:
                return float(h.iloc[-1])
        else:
            fi = yf.Ticker(ticker).fast_info
            last = fi.get("last_price", None)
            if last is not None and not np.isnan(last):
                return float(last)
    except Exception:
        pass
    return None

# -------------------------------
# Function: Intraday chart + AI trading signal
# -------------------------------
def intraday_chart_with_signal(ticker: str):
    try:
        df = yf.download(ticker, period="5d", interval="5m", auto_adjust=False, progress=False)
        if df.empty:
            return None, "No intraday data available."

        today = datetime.datetime.now().date()
        df = df[df.index.date == today]
        if df.empty:
            return None, "No intraday candles for today (market may be closed)."

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

        if price > ema9 > ema21 and rsi < 70:
            signal = "BUY ✅ (bullish momentum)"
        elif price < ema21 or rsi > 70:
            signal = "SELL ❌ (weakening or overbought)"
        else:
            signal = "HOLD 🤝 (sideways)"

        # Candlestick chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candles",
            increasing_line_color="green",
            decreasing_line_color="red",
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], line=dict(color="blue", width=1.5), name="EMA 9"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], line=dict(color="orange", width=1.5), name="EMA 21"))

        fig.update_layout(
            title=f"{ticker} Intraday (5m) Candlestick",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=600,
            yaxis_title="Price ($)"
        )

        return fig, signal
    except Exception as e:
        return None, f"Error: {e}"

# -------------------------------
# Function: News + Sentiment
# -------------------------------
def get_news_headlines(ticker: str, max_items: int = 5):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        r = requests.get(url, timeout=5).json()
        news = []
        if "news" in r:
            for item in r["news"][:max_items]:
                title = item.get("title")
                link = item.get("link")
                if title and link:
                    news.append((title, link))
        return news
    except Exception:
        return []

def analyze_sentiment(texts):
    if not texts:
        return "Neutral 🟡", []
    scores = []
    for t in texts:
        polarity = TextBlob(t).sentiment.polarity
        if polarity > 0.15:
            scores.append(1)
        elif polarity < -0.15:
            scores.append(-1)
        else:
            scores.append(0)
    avg = np.mean(scores)
    if avg > 0:
        sentiment = "Positive 🟢"
    elif avg < 0:
        sentiment = "Negative 🔴"
    else:
        sentiment = "Neutral 🟡"
    cumulative = np.cumsum(scores).tolist()
    return sentiment, cumulative

def sentiment_badge(sentiment):
    return f"**{sentiment}**"

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
    for _, row in enumerate(portfolio_rows):
        t = str(row["Ticker"]).upper().strip()
        qty = float(row["Quantity"])  # keep decimals (crypto friendly)
        cost = float(row["Cost"])
        try:
            lp = _live_price_now(t)
            if lp is None:
                lp = _to_float(yf.Ticker(t).history(period="5d", interval="1d")["Close"].iloc[-1])

            value = lp * qty
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
                why = "Price above 200-day MA."
            else:
                action = "HOLD 🤝"
                why = "Within normal range."

            headlines = get_news_headlines(t)
            sentiment, cumulative_trend = analyze_sentiment([h[0] for h in headlines])

            results.append({
                "Ticker": t,
                "Quantity": round(qty, 8),  # 👈 up to 8 decimals for BTC etc
                "Cost Basis": round(cost, 2),
                "Live Price": round(lp, 2),
                "Value ($)": round(value, 2),
                "P/L ($)": round(pnl, 2),
                "P/L (%)": f"{pnl_pct*100:,.2f}%" if pnl_pct == pnl_pct else "N/A",
                "Action": action,
                "Reason": why,
                "News Sentiment": sentiment_badge(sentiment),
                "Headlines": headlines,
                "Sentiment Trend": cumulative_trend
            })
        except Exception:
            continue
    return pd.DataFrame(results)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Controls")
universe = st.sidebar.text_input("Enter tickers for Trending (comma separated):",
                                 "AAPL, MSFT, TSLA, AMD, NVDA, RGTI, BTC-USD")

# -------------------------------
# Stock Checker
# -------------------------------
st.subheader("🔎 Stock Checker – AI Intraday Signal")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA, BTC-USD):")
if ticker:
    fig, signal = intraday_chart_with_signal(ticker.upper())
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"🤖 AI Trading Signal: {signal}")
    else:
        st.warning(signal)

# -------------------------------
# Portfolio Section (CSV Upload or Manual Input)
# -------------------------------
st.subheader("💼 Portfolio Analyzer")

uploaded_file = st.file_uploader("📂 Upload your portfolio CSV (columns: Stock, Quantity, Price Purchased)", type=["csv"])

portfolio_rows = []
if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    for _, row in df_csv.iterrows():
        portfolio_rows.append({"Ticker": str(row["Stock"]).upper(), 
                               "Quantity": float(row["Quantity"]), 
                               "Cost": float(row["Price Purchased"])})
else:
    st.info("Or manually input your portfolio below:")
    with st.form("portfolio_form"):
        num_stocks = st.number_input("How many stocks do you want to add?", min_value=1, max_value=15, value=3)
        for i in range(num_stocks):
            c1, c2, c3 = st.columns([2, 1, 1])
            ticker = c1.text_input(f"Ticker {i+1}", value="AAPL" if i == 0 else "")
            qty = c2.number_input(f"Quantity {i+1}", min_value=0.0, step=0.00000001, format="%.8f", value=10.0 if i == 0 else 0.0)
            cost = c3.number_input(f"Purchase Price {i+1}", min_value=0.0, value=150.0 if i == 0 else 0.0)
            if ticker:
                portfolio_rows.append({"Ticker": ticker.upper(), "Quantity": qty, "Cost": cost})
        submitted = st.form_submit_button("Analyze Portfolio")

# -------------------------------
# Portfolio Analysis
# -------------------------------
if portfolio_rows:
    df_portfolio = analyze_portfolio(portfolio_rows)
    if not df_portfolio.empty:
        st.dataframe(df_portfolio.drop(columns=["Headlines","Sentiment Trend"]), hide_index=True, use_container_width=True)

        # Totals
        total_value = float(np.nansum(df_portfolio["Live Price"] * df_portfolio["Quantity"]))
        total_pl = float(np.nansum(df_portfolio["P/L ($)"]))
        c1, c2 = st.columns(2)
        c1.metric("📊 Portfolio Value", f"${total_value:,.2f}")
        c2.metric("💰 Total P/L", f"${total_pl:,.2f}")

        # Charts
        st.markdown("### 📊 Portfolio Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = go.Figure(go.Bar(
                x=df_portfolio["Ticker"],
                y=df_portfolio["P/L ($)"],
                marker=dict(color=np.where(df_portfolio["P/L ($)"] >= 0, "green", "red"))
            ))
            fig_bar.update_layout(title="Profit/Loss per Stock", yaxis_title="P/L ($)")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_pie = go.Figure(go.Pie(labels=df_portfolio["Ticker"], values=df_portfolio["Value ($)"], hole=0.4))
            fig_pie.update_layout(title="Portfolio Allocation by Value")
            st.plotly_chart(fig_pie, use_container_width=True)

        # News & Sentiment
        st.markdown("### 📰 Latest News & Sentiment Trends")
        for _, row in df_portfolio.iterrows():
            st.markdown(f"**{row['Ticker']} — {row['News Sentiment']}**")
            headlines = row["Headlines"]
            if isinstance(headlines, list) and headlines:
                for title, link in headlines:
                    st.markdown(f"- [{title}]({link})")
            else:
                st.write("No recent headlines.")
            trend = row["Sentiment Trend"]
            if trend:
                trend_df = pd.DataFrame({"Momentum": trend, "Headline #": range(1, len(trend)+1)})
                st.line_chart(trend_df.set_index("Headline #"))

st.caption("⚠️ Educational purposes only. This is not investment advice.")
