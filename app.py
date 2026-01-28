import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="BTC Diversification Dashboard", layout="wide")


# -----------------------------
# Data loading utilities
# -----------------------------
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


# -----------------------------
# Finance helpers
# -----------------------------
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


def wide_to_long(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert a wide dataframe with a DatetimeIndex into long format for plotly express."""
    out = df_wide.copy()
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out = out.melt(id_vars="timestamp", var_name="Portfolio", value_name=value_name)
    return out


# -----------------------------
# App
# -----------------------------
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

# Handle case where Streamlit returns a single date instead of a tuple
if isinstance(date_range, tuple) or isinstance(date_range, list):
    start_date, end_date = date_range
else:
    start_date, end_date = date_range, date_range

roll_window = st.sidebar.slider("Rolling window (days)", 30, 252, 90, 10)
w_btc = st.sidebar.slider("Bitcoin weight", 0.0, 0.5, 0.10, 0.01)
w_spy = 1.0 - w_btc
rf_annual = st.sidebar.number_input("Risk-free rate (annual)", 0.0, 0.10, 0.02, 0.005)

# Filter data
df_f = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)].copy()
rets = compute_returns(df_f)

if rets.empty or len(rets) < roll_window + 5:
    st.warning("Not enough data in the selected date range. Expand the range or reduce the rolling window.")
    st.stop()

# Portfolios
spy_only = port_rets(rets, 1.0, 0.0)
mix = port_rets(rets, w_spy, w_btc)

mix_label = f"{int(w_spy*100)}/{int(w_btc*100)} SPY/BTC"

# Top metrics
corr = float(rets["BTC Close"].corr(rets["SPY Close"]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC–SPY Correlation (returns)", f"{corr:.2f}")
c2.metric("SPY Only Vol (ann.)", f"{ann_vol(spy_only):.2%}")
c3.metric("Mix Vol (ann.)", f"{ann_vol(mix):.2%}")
c4.metric("Mix Sharpe", f"{sharpe(mix, rf_annual):.2f}")


# -----------------------------
# Chart 1: Cumulative Growth (keep)
# -----------------------------
st.subheader("Cumulative Growth of $1")

growth = pd.DataFrame(
    {
        "SPY Only": (1 + spy_only).cumprod(),
        mix_label: (1 + mix).cumprod(),
    },
    index=rets.index
)

growth_long = wide_to_long(growth, value_name="Growth")

fig_growth = px.line(
    growth_long,
    x="timestamp",
    y="Growth",
    color="Portfolio",
    title=None
)
fig_growth.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value")
st.plotly_chart(fig_growth, use_container_width=True)


# -----------------------------
# Chart 2: Rolling Annualized Volatility (keep + fix)
# -----------------------------
st.subheader("Rolling Annualized Volatility")

rolling_vol = pd.DataFrame(
    {
        "SPY Only": spy_only.rolling(roll_window).std() * np.sqrt(252),
        mix_label: mix.rolling(roll_window).std() * np.sqrt(252),
    },
    index=rets.index
).dropna()

if rolling_vol.empty:
    st.warning("Rolling window too large for the selected date range. Reduce the window or expand the date range.")
    st.stop()

rolling_vol_long = wide_to_long(rolling_vol, value_name="Annualized Vol")

fig_vol = px.line(
    rolling_vol_long,
    x="timestamp",
    y="Annualized Vol",
    color="Portfolio",
    title=None
)
fig_vol.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility")
st.plotly_chart(fig_vol, use_container_width=True)


# -----------------------------
# Chart 3: Rolling Correlation (keep)
# -----------------------------
st.subheader("Rolling Correlation (BTC vs SPY)")

rolling_corr = rets["BTC Close"].rolling(roll_window).corr(rets["SPY Close"]).dropna()

if rolling_corr.empty:
    st.warning("Rolling correlation could not be computed for this window/range. Try a smaller window.")
    st.stop()

corr_df = rolling_corr.to_frame("Rolling Corr").reset_index().rename(columns={"index": "timestamp"})

fig_corr = px.line(
    corr_df,
    x="timestamp",
    y="Rolling Corr",
    title=None
)
fig_corr.update_layout(xaxis_title="Date", yaxis_title="Rolling Correlation")
st.plotly_chart(fig_corr, use_container_width=True)


# -----------------------------
# Summary Table (keep)
# -----------------------------
st.subheader("Summary Table")

summary = pd.DataFrame(
    {
        "Annualized Return": [ann_ret(spy_only), ann_ret(mix)],
        "Annualized Vol": [ann_vol(spy_only), ann_vol(mix)],
        "Sharpe": [sharpe(spy_only, rf_annual), sharpe(mix, rf_annual)],
    },
    index=["SPY Only", "SPY/BTC Mix"]
)

st.dataframe(
    summary.style.format(
        {"Annualized Return": "{:.2%}", "Annualized Vol": "{:.2%}", "Sharpe": "{:.2f}"}
    )
)
