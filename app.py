import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="BTC Diversification Dashboard", layout="wide")

# -----------------------------
# Data fetch + helpers
# -----------------------------
@st.cache_data(ttl=60 * 60)  # cache for 1 hour
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

def get_api_key() -> str:
    # Prefer Streamlit secrets, fallback to env var
    if hasattr(st, "secrets") and "ALPHAVANTAGE_API_KEY" in st.secrets:
        return st.secrets["ALPHAVANTAGE_API_KEY"]
    return os.getenv("ALPHAVANTAGE_API_KEY")

@st.cache_data(ttl=60 * 60)
def load_btc_spy(api_key: str) -> pd.DataFrame:
    # BTC daily (AlphaVantage DIGITAL_CURRENCY_DAILY)
    btc_url = (
        "https://www.alphavantage.co/query"
        f"?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR"
        f"&apikey={api_key}&outputsize=full&datatype=csv"
    )

    # SPY daily adjusted
    spy_url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY"
        f"&apikey={api_key}&outputsize=full&datatype=csv"
    )

    btc = fetch_csv(btc_url)
    spy = fetch_csv(spy_url)

    # Parse dates
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], errors="coerce")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], errors="coerce")

    # Identify BTC close column (AlphaVantage crypto CSV uses e.g. "close (EUR)")
    btc_close_col = None
    for c in btc.columns:
        if "close" in c.lower():
            btc_close_col = c
            break
    if btc_close_col is None:
        raise ValueError("Could not find BTC close column in AlphaVantage response.")

    btc = btc[["timestamp", btc_close_col]].rename(columns={btc_close_col: "BTC Close"})
    spy = spy[["timestamp", "close"]].rename(columns={"close": "SPY Close"})

    # Merge and clean
    df = pd.merge(spy, btc, on="timestamp", how="inner").dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Force numeric (important)
    df["SPY Close"] = pd.to_numeric(df["SPY Close"], errors="coerce")
    df["BTC Close"] = pd.to_numeric(df["BTC Close"], errors="coerce")
    df = df.dropna(subset=["SPY Close", "BTC Close"])

    return df

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    # Simple daily returns
    rets = df[["SPY Close", "BTC Close"]].pct_change()
    return rets.dropna()

def port_rets(rets: pd.DataFrame, w_spy: float, w_btc: float) -> pd.Series:
    return w_spy * rets["SPY Close"] + w_btc * rets["BTC Close"]

def a
