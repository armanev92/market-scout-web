
import streamlit as st

# --- Fake stock evaluator for demo ---
def check_stock(ticker: str):
    ticker = ticker.upper()
    if ticker in ["AAPL", "AMD", "NVDA", "MSFT", "TSLA"]:
        return "✅ PASS", f"{ticker} shows strong momentum and fundamentals."
    else:
        return "❌ FAIL", f"{ticker} does not meet the criteria right now."

# --- Streamlit UI ---
st.set_page_config(page_title="Market Scout GPT", layout="wide")
st.title("📈 Market Scout – PASS/FAIL Stock Checker")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, AMD, TSLA):")

if ticker:
    result, reason = check_stock(ticker)
    st.subheader(result)
    st.write(reason)
