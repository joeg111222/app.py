import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="BTC Diversification Dashboard", layout="wide")

@st.cache_data(ttl=60 * 60)
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

def get_api_key() -> str:
    if hasattr(st, "secrets") and "ALPHAVANTAGE_API_KEY" in st.secrets:
        return st.secrets["ALPHAVANTAGE_API_KEY"]
    return os.getenv("ALPHAVANTAGE_API_KEY")

@st.cache_data(ttl=60 * 60)
def load_btc_spy(api_key: str) -> pd.DataFrame:
    btc_url = (
        "https://www.alphavantage.co/query"
        f"?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR"
        f"&apikey={api_key}&outputsize=full&datatype=csv"
    )
    spy_url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY"
        f"&apikey={api_key}&outputsize=full&datatype=csv"
    )

    btc = fetch_csv(btc_url)
    spy = fetch_csv(spy_url)

    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    spy["timestamp"] = pd.to_datetime(spy["timestamp"])

    btc_close_col = None
    for c in btc.columns:
        if "close" in c.lower():
            btc_close_col = c
            break
    if btc_close_col is None:
        raise ValueError("Could not find BTC close column in AlphaVantage response.")

    btc = btc[["timestamp", btc_close_col]].rename(columns={btc_close_col: "BTC Close"})
    spy = spy[["timestamp", "close"]].rename(columns={"close": "SPY Close"})

    df = pd.merge(spy, btc, on="timestamp", how="inner").sort_values("timestamp")
    df = df.set_index("timestamp")
    return df

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df[["SPY Close", "BTC Close"]].pct_change().dropna()

def port_rets(rets: pd.DataFrame, w_spy: float, w_btc: float) -> pd.Series:
    return w_spy * rets["SPY Close"] + w_btc * rets["BTC Close"]

def ann_vol(x: pd.Series) -> float:
    return float(x.std() * np.sqrt(252))

def ann_ret(x: pd.Series) -> float:
    return float(x.mean() * 252)

def sharpe(x: pd.Series, rf_annual: float) -> float:
    rf_daily = rf_annual / 252
    excess = x - rf_daily
    denom = x.std()
    return np.nan if denom == 0 or np.isnan(denom) else float((excess.mean() / denom) * np.sqrt(252))

st.title("Bitcoin vs SPY: Diversification Dashboard")

api_key = get_api_key()
if not api_key:
    st.error("Missing AlphaVantage API key. Add it in Streamlit Cloud > Secrets as ALPHAVANTAGE_API_KEY.")
    st.stop()

df = load_btc_spy(api_key)

st.sidebar.header("Controls")

min_date = df.index.min().date()
max_date = df.index.max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
start_date, end_date = date_range

roll_window = st.sidebar.slider("Rolling window (days)", 30, 252, 90, 10)
w_btc = st.sidebar.slider("Bitcoin weight", 0.0, 0.5, 0.10, 0.01)
w_spy = 1.0 - w_btc
rf_annual = st.sidebar.number_input("Risk-free rate (annual)", 0.0, 0.10, 0.02, 0.005)

df_f = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)].copy()
rets = compute_returns(df_f)

spy_only = port_rets(rets, 1.0, 0.0)
mix = port_rets(rets, w_spy, w_btc)

corr = float(rets["BTC Close"].corr(rets["SPY Close"]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC–SPY Correlation (returns)", f"{corr:.2f}")
c2.metric("SPY Only Vol (ann.)", f"{ann_vol(spy_only):.2%}")
c3.metric("Mix Vol (ann.)", f"{ann_vol(mix):.2%}")
c4.metric("Mix Sharpe", f"{sharpe(mix, rf_annual):.2f}")

st.subheader("Cumulative Growth of $1")
growth = pd.DataFrame({
    "SPY Only": (1 + spy_only).cumprod(),
    f"{int(w_spy*100)}/{int(w_btc*100)} SPY/BTC": (1 + mix).cumprod(),
})
st.plotly_chart(px.line(growth, x=growth.index, y=growth.columns), use_container_width=True)

st.subheader("Rolling Annualized Volatility")
rolling_vol = pd.DataFrame({
    "SPY Only": spy_only.rolling(roll_window).std() * np.sqrt(252),
    f"{int(w_spy*100)}/{int(w_btc*100)} SPY/BTC": mix.rolling(roll_window).std() * np.sqrt(252),
}).dropna()
st.plotly_chart(px.line(rolling_vol, x=rolling_vol.index, y=rolling_vol.columns), use_container_width=True)

st.subheader("Rolling Correlation")
rolling_corr = rets["BTC Close"].rolling(roll_window).corr(rets["SPY Close"]).dropna()
st.plotly_chart(px.line(x=rolling_corr.index, y=rolling_corr.values, labels={"x": "Date", "y": "Rolling Corr"}), use_container_width=True)

st.subheader("Summary Table")
summary = pd.DataFrame({
    "Annualized Return": [ann_ret(spy_only), ann_ret(mix)],
    "Annualized Vol": [ann_vol(spy_only), ann_vol(mix)],
    "Sharpe": [sharpe(spy_only, rf_annual), sharpe(mix, rf_annual)],
}, index=["SPY Only", "SPY/BTC Mix"])
st.dataframe(summary.style.format({"Annualized Return": "{:.2%}", "Annualized Vol": "{:.2%}", "Sharpe": "{:.2f}"}))
