import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Market Scout Pro", page_icon="📈", layout="wide")
st.title("📈 Market Scout – Pro Dashboard")

# -------------------------------
# Helpers
# -------------------------------
def _to_float(x) -> float:
    """Safely convert the last element of a Series/array/scalar to float."""
    if isinstance(x, (pd.Series, pd.Index, list, tuple, np.ndarray)):
        return float(pd.Series(x).dropna().iloc[-1])
    return float(x)

def _live_price_now(ticker: str) -> float | None:
    """Try fast live price, then 1m history fallback."""
    try:
        fi = yf.Ticker(ticker).fast_info
        last = fi.get("last_price", None)
        if last is not None and not np.isnan(last):
            return float(last)
    except Exception:
        pass
    # Fallback to 1-minute bars from today
    try:
        h = yf.Ticker(ticker).history(period="1d", interval="1m")["Close"].dropna()
        if not h.empty:
            return float(h.iloc[-1])
    except Exception:
        pass
    return None

# -------------------------------
# Function: Check stock with MA
# -------------------------------
def check_stock(ticker: str, ma_days: int):
    try:
        data = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            return "FAIL", f"No data found for {ticker}", None

        ma_col = f"{ma_days}_MA"
        data[ma_col] = data["Close"].rolling(ma_days).mean()

        # ✅ Extract scalars (floats) to avoid Series ambiguity
        price = _to_float(data["Close"].iloc[-1])
        ma_value_series = data[ma_col].dropna()
        if ma_value_series.empty:
            return "FAIL", f"Not enough data to compute {ma_days}-day MA for {ticker}.", data
        ma_value = _to_float(ma_value_series.iloc[-1])

        if price > ma_value:
            return "PASS", f"{ticker} is trading above its {ma_days}-day moving average ({price:.2f} > {ma_value:.2f}).", data
        else:
            return "FAIL", f"{ticker} is trading below its {ma_days}-day moving average ({price:.2f} < {ma_value:.2f}).", data

    except Exception as e:
        return "FAIL", f"Error checking {ticker}: {e}", None

# -------------------------------
# Function: Trending stocks
# -------------------------------
def get_trending(tickers: list[str]) -> pd.DataFrame:
    try:
        close = yf.download(tickers, period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"]
        # Normalize shape (Series -> DataFrame)
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])

        if close.shape[0] < 5:
            return pd.DataFrame({"Note": ["Not enough days to compute 5d change"]})

        changes = (close.iloc[-1] / close.iloc[-5] - 1.0) * 100.0
        out = changes.sort_values(ascending=False).to_frame("5d % Change").reset_index()
        out.rename(columns={"index": "Ticker"}, inplace=True)
        return out
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

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
            # Live price
            lp = _live_price_now(t)
            if lp is None:
                # fallback to last daily close
                lp = _to_float(yf.Ticker(t).history(period="5d", interval="1d")["Close"].iloc[-1])

            pnl = (lp - cost) * qty
            pnl_pct = (lp - cost) / cost if cost else np.nan

            # Trend context (200MA)
            hist = yf.download(t, period="6mo", interval="1d", auto_adjust=True, progress=False)
            if not hist.empty:
                ma200_ser = hist["Close"].rolling(200).mean().dropna()
                ma200 = _to_float(ma200_ser.iloc[-1]) if not ma200_ser.empty else np.nan
            else:
                ma200 = np.nan

            # Simple rule-based suggestion
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
ma_days = st.sidebar.selectbox("Choose moving average window:", [20, 50, 200], index=1)
universe = st.sidebar.text_input(
    "Enter tickers for Trending (comma separated):",
    "AAPL, MSFT, TSLA, AMD, NVDA, RGTI"
)

# -------------------------------
# Stock Checker
# -------------------------------
st.subheader("🔎 Stock Checker – PASS/FAIL")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA):")

if ticker:
    # Live price (outside chart AND used inside chart)
    live_price = _live_price_now(ticker)

    result, reason, data = check_stock(ticker.upper(), ma_days)
    if result == "PASS":
        st.success(f"✅ {result}")
    else:
        st.error(f"❌ {result}")
    st.write(reason)

    if live_price is not None:
        st.metric(label=f"💲 Live Price ({ticker.upper()})", value=f"${live_price:.2f}")

    if data is not None and not data.empty:
        ma_col = f"{ma_days}_MA"
        ma_series = data[ma_col].dropna()
        if not ma_series.empty:
            ma_value = _to_float(ma_series.iloc[-1])
        else:
            ma_value = np.nan

        # ---- Matplotlib chart with live price & % diff ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data["Close"], label="Price (Daily Close)", color="blue")
        ax.plot(data.index, data[ma_col], label=f"{ma_days}-Day MA", color="orange")

        # Live price overlay + % diff vs MA
        if live_price is not None and not np.isnan(ma_value):
            pct_diff = ((live_price - ma_value) / ma_value) * 100.0
            ax.axhline(live_price, color="red", linestyle="--", linewidth=1.2)
            ax.scatter(data.index[-1], live_price, color="red", zorder=5)
            label_color = "green" if pct_diff >= 0 else "red"
            ax.text(
                data.index[-1],
                live_price,
                f" Live: ${live_price:.2f} ({pct_diff:+.2f}%)",
                color=label_color,
                fontsize=9,
                verticalalignment="bottom"
            )

        ax.set_title(f"{ticker.upper()} Price vs {ma_days}-Day MA")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

# -------------------------------
# Trending Section
# -------------------------------
st.subheader("🔥 Top Trending Stocks (5d % Change)")
if universe.strip():
    tickers_list = [t.strip().upper() for t in universe.split(",") if t.strip()]
    trending = get_trending(tickers_list)

    if not trending.empty and "5d % Change" in trending.columns:
        def color_format(val):
            try:
                return f"color: {'green' if val > 0 else 'red'}; font-weight: bold;"
            except Exception:
                return ""
        st.dataframe(trending.style.applymap(color_format, subset=["5d % Change"]))
    else:
        st.dataframe(trending)  # show note or error if present

# -------------------------------
# Portfolio Section
# -------------------------------
st.subheader("💼 Portfolio Analyzer")
st.markdown("Enter your positions below (Ticker, Quantity, Cost Basis). One per line, e.g.: `AAPL,10,150`")

with st.form("portfolio_form"):
    tickers_input = st.text_area(
        "Portfolio (Ticker,Quantity,Cost)",
        "AAPL,10,150\nTSLA,5,700\nRGTI,20,15"
    )
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
        # Totals
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
st.caption("Educational purposes only. This is not investment advice.")
