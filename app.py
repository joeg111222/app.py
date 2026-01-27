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
@st.cache_data(ttl=60 * 60)  # cache for 1 hour to reduce API calls / rate limits
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

def get_api_key() -> str:
    # Streamlit Cloud secrets (preferred) or local env var fallback
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

    # Parse timestamps
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], errors="coerce")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], errors="coerce")

    # Find BTC close column (often named like "close (EUR)")
    btc_close_col = None
    for c in btc.columns:
        if "close" in c.lower():
            btc_close_col = c
            break
    if btc_close_col is None:
        raise ValueError("Could not find BTC close column in AlphaVantage response.")

    btc = btc[["timestamp", btc_close_col]].rename(columns={btc_close_col: "BTC Close"})
    spy = spy[["timestamp", "close"]].rename(columns={"close": "SPY Close"})

    # Merge, sort, set index
    df = pd.merge(spy, btc, on="timestamp", how="inner")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    # Force numeric
    df["SPY Close"] = pd.to_numeric(df["SPY Close"], errors="coerce")
    df["BTC Close"] = pd.to_numeric(df["BTC Close"], errors="coerce")
    df = df.dropna(subset=["SPY Close", "BTC Close"])

    return df

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    # Daily simple returns
    rets = df[["SPY Close", "BTC Close"]].pct_change()
    return rets.dropna()

def port_rets(rets: pd.DataFrame, w_spy: float, w_btc: float) -> pd.Series:
    return w_spy * rets["SPY Close"] + w_btc * rets["BTC Close"]

def ann_vol(x: pd.Series) -> float:
    return float(x.std() * np.sqrt(252))

def ann_ret(x: pd.Series) -> float:
    return float(x.mean() * 252)

def sharpe(x: pd.Series, rf_annual: float) -> float:
    rf_daily = rf_annual / 252.0
    excess = x - rf_daily
    denom = x.std()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float((excess.mean() / denom) * np.sqrt(252))

# -----------------------------
# App UI
# -----------------------------
st.title("Bitcoin vs SPY: Diversification Dashboard")
st.caption("Interactive portfolio view: correlation, rolling risk, and diversification impact (no price forecasting).")

api_key = get_api_key()
if not api_key:
    st.error("Missing AlphaVantage API key. Add it in Streamlit Cloud → Settings → Secrets as ALPHAVANTAGE_API_KEY.")
    st.stop()

df = load_btc_spy(api_key)

# Sidebar controls
st.sidebar.header("Controls")

min_date = df.index.min().date()
max_date = df.index.max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Handle Streamlit date_input behavior safely
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

roll_window = st.sidebar.slider("Rolling window (days)", min_value=30, max_value=252, value=90, step=10)
w_btc = st.sidebar.slider("Bitcoin weight", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
rf_annual = st.sidebar.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)

w_spy = 1.0 - w_btc

# Filter data (must happen before computing returns so sliders update results)
df_f = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)].copy()

# Ensure numeric on filtered data as well (extra safety)
df_f["SPY Close"] = pd.to_numeric(df_f["SPY Close"], errors="coerce")
df_f["BTC Close"] = pd.to_numeric(df_f["BTC Close"], errors="coerce")
df_f = df_f.dropna(subset=["SPY Close", "BTC Close"])

rets = compute_returns(df_f)

# Guard against too-short samples (common if user picks a tiny date range)
if len(rets) < roll_window + 5:
    st.warning("Selected date range is too short for the chosen rolling window. Expand the date range or reduce the rolling window.")
    st.stop()

spy_only = port_rets(rets, 1.0, 0.0)
mix = port_rets(rets, w_spy, w_btc)

# Metrics
corr = float(rets["BTC Close"].corr(rets["SPY Close"]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC–SPY Correlation (returns)", f"{corr:.2f}")
c2.metric("SPY Only Vol (ann.)", f"{ann_vol(spy_only):.2%}")
c3.metric("Mix Vol (ann.)", f"{ann_vol(mix):.2%}")
c4.metric("Mix Sharpe", f"{sharpe(mix, rf_annual):.2f}")

# -----------------------------
# Cumulative Growth of $1
# -----------------------------
st.subheader("Cumulative Growth of $1")

growth = pd.DataFrame({
    "Date": spy_only.index,
    "SPY Only": (1 + spy_only).cumprod().values,
    f"{int(w_spy*100)}/{int(w_btc*100)} SPY/BTC": (1 + mix).cumprod().values,
})

fig_growth = px.line(
    growth,
    x="Date",
    y=[c for c in growth.columns if c != "Date"],
)
st.plotly_chart(fig_growth, use_container_width=True)

# -----------------------------
# Rolling Annualized Volatility
# -----------------------------
st.subheader("Rolling Annualized Volatility")

rolling_vol = pd.DataFrame({
    "Date": spy_only.index,
    "SPY Only": (spy_only.rolling(roll_window).std() * np.sqrt(252)).values,
    f"{int(w_spy*100)}/{int(w_btc*100)} SPY/BTC": (mix.rolling(roll_window).std() * np.sqrt(252)).values,
}).dropna()

fig_rvol = px.line(
    rolling_vol,
    x="Date",
    y=[c for c in rolling_vol.columns if c != "Date"],
    title=f"Rolling Annualized Volatility ({roll_window}-day window)"
)
st.plotly_chart(fig_rvol, use_container_width=True)

# -----------------------------
# Rolling Correlation
# -----------------------------
st.subheader("Rolling Correlation (BTC vs SPY)")

rolling_corr = rets["BTC Close"].rolling(roll_window).corr(rets["SPY Close"]).dropna()
rolling_corr_df = pd.DataFrame({"Date": rolling_corr.index, "Rolling Corr": rolling_corr.values})

fig_rcorr = px.line(
    rolling_corr_df,
    x="Date",
    y="Rolling Corr",
    title=f"Rolling Correlation ({roll_window}-day window)"
)
st.plotly_chart(fig_rcorr, use_container_width=True)

# -----------------------------
# Summary Table
# -----------------------------
st.subheader("Summary Table")

summary = pd.DataFrame({
    "Annualized Return": [ann_ret(spy_only), ann_ret(mix)],
    "Annualized Vol": [ann_vol(spy_only), ann_vol(mix)],
    "Sharpe (rf input)": [sharpe(spy_only, rf_annual), sharpe(mix, rf_annual)],
}, index=["SPY Only", f"{int(w_spy*100)}/{int(w_btc*100)} SPY/BTC"])

st.dataframe(summary.style.format({
    "Annualized Return": "{:.2%}",
    "Annualized Vol": "{:.2%}",
    "Sharpe (rf input)": "{:.2f}",
}))
