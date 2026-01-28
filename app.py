import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="BTC Diversification Dashboard", layout="wide")

# Plotly dark baseline so charts match the theme
px.defaults.template = "plotly_dark"

# -----------------------------
# Blade Runner / Cyberpunk CSS
# -----------------------------
st.markdown(
    """
<style>
/* Global */
:root{
  --bg0:#070A12;
  --bg1:#0B1020;
  --card:#0E1630;
  --grid:#1B2A4A;
  --text:#D7E2FF;
  --muted:#8EA4D8;

  --neon1:#00F5FF;  /* cyan */
  --neon2:#FF2BD6;  /* magenta */
  --neon3:#7CFF6B;  /* acid green */
  --warn:#FFB020;
}

html, body, [class*="css"]  {
  background: radial-gradient(1200px 800px at 20% 10%, rgba(0,245,255,0.08), transparent 60%),
              radial-gradient(900px 700px at 80% 20%, rgba(255,43,214,0.08), transparent 60%),
              radial-gradient(900px 700px at 50% 90%, rgba(124,255,107,0.06), transparent 60%),
              var(--bg0);
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

/* Tighten top padding */
.block-container{
  padding-top: 1.2rem;
  padding-bottom: 2.0rem;
}

/* Headings */
h1, h2, h3 {
  letter-spacing: 0.5px;
}
h1{
  text-shadow: 0 0 12px rgba(0,245,255,0.25);
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.98), rgba(7,10,18,0.98));
  border-right: 1px solid rgba(0,245,255,0.15);
}
section[data-testid="stSidebar"] * {
  color: var(--text);
}

/* Cards (Metrics) */
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(14,22,48,0.96), rgba(11,16,32,0.96));
  border: 1px solid rgba(0,245,255,0.18);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow:
    0 0 18px rgba(0,245,255,0.12),
    0 0 24px rgba(255,43,214,0.08);
}
div[data-testid="stMetricLabel"]{
  color: var(--muted) !important;
  font-size: 0.85rem !important;
}
div[data-testid="stMetricValue"]{
  color: var(--text) !important;
  font-size: 1.35rem !important;
}
div[data-testid="stMetricDelta"]{
  color: var(--neon3) !important;
}

/* Tabs */
div[data-baseweb="tab-list"]{
  gap: 6px;
}
button[data-baseweb="tab"]{
  background: rgba(14,22,48,0.55);
  border: 1px solid rgba(0,245,255,0.18);
  border-radius: 14px;
  padding: 8px 14px;
  box-shadow: 0 0 12px rgba(0,245,255,0.08);
}
button[data-baseweb="tab"][aria-selected="true"]{
  background: rgba(14,22,48,0.85);
  border: 1px solid rgba(255,43,214,0.35);
  box-shadow:
    0 0 14px rgba(255,43,214,0.14),
    0 0 18px rgba(0,245,255,0.10);
}

/* Dataframe */
div[data-testid="stDataFrame"]{
  border: 1px solid rgba(0,245,255,0.16);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 0 18px rgba(0,245,255,0.08);
}

/* Subtle divider */
hr{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,245,255,0.35), transparent);
  margin: 1rem 0;
}

/* Hide Streamlit default menu/footer for cleaner cyberpunk look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True
)

# -----------------------------
# Data + helpers
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

    btc["timestamp"] = pd.to_datetime(btc["timestamp"], errors="coerce")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], errors="coerce")

    # Find BTC close column (often "close (EUR)")
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

    # Force numeric
    df["SPY Close"] = pd.to_numeric(df["SPY Close"], errors="coerce")
    df["BTC Close"] = pd.to_numeric(df["BTC Close"], errors="coerce")
    df = df.dropna(subset=["SPY Close", "BTC Close"])

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
    rf_daily = rf_annual / 252.0
    excess = x - rf_daily
    denom = x.std()
    if np.isnan(denom) or denom < 1e-8:
        return np.nan
    return float((excess.mean() / denom) * np.sqrt(252))


def to_long(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    out = df_wide.copy()
    out = out.reset_index().rename(columns={"timestamp": "Date"})
    out = out.melt(id_vars="Date", var_name="Series", value_name=value_name)
    return out


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div style="padding: 10px 0 2px 0;">
  <div style="font-size: 34px; font-weight: 800; letter-spacing: 1px;">
    BITCOIN × SPY — DIVERSIFICATION DASHBOARD
  </div>
  <div style="color: #8EA4D8; font-size: 13px; margin-top: 6px;">
    Cyberpunk view of portfolio behavior: growth, time-varying risk, and correlation regimes.
  </div>
</div>
""",
    unsafe_allow_html=True
)

api_key = get_api_key()
if not api_key:
    st.error("Missing AlphaVantage API key. Add it in Streamlit Cloud → Secrets as ALPHAVANTAGE_API_KEY.")
    st.stop()

df = load_btc_spy(api_key)

# -----------------------------
# Controls (Sidebar)
# -----------------------------
st.sidebar.markdown("### Controls")

min_date = df.index.min().date()
max_date = df.index.max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Defensive handling in case Streamlit returns a single date
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

roll_window = st.sidebar.slider("Rolling window (days)", 30, 252, 90, 10)
w_btc = st.sidebar.slider("Bitcoin weight", 0.0, 0.5, 0.10, 0.01)
w_spy = 1.0 - w_btc

# Risk-free rate: reasonable range, 0.01 steps (your request)
rf_annual = st.sidebar.number_input(
    "Risk-free rate (annual)",
    min_value=0.00,
    max_value=0.10,
    value=0.02,
    step=0.01
)

st.sidebar.markdown(
    """
<div style="margin-top: 10px; color: #8EA4D8; font-size: 12px; line-height: 1.35;">
Rolling metrics use the last <b>N</b> trading days to estimate near-term risk.  
Bigger windows smooth noise but reduce early data points.
</div>
""",
    unsafe_allow_html=True
)

# -----------------------------
# Filter + compute
# -----------------------------
df_f = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)].copy()
rets = compute_returns(df_f)

if rets.empty or len(rets) < roll_window + 5:
    st.warning("Not enough data for this date range / rolling window. Expand the range or reduce the window.")
    st.stop()

spy_only = port_rets(rets, 1.0, 0.0)
mix = port_rets(rets, w_spy, w_btc)
mix_label = f"{int(w_spy*100)}/{int(w_btc*100)} SPY/BTC"

corr = float(rets["BTC Close"].corr(rets["SPY Close"]))

# -----------------------------
# KPI row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("BTC–SPY Corr (returns)", f"{corr:.2f}")
k2.metric("SPY Vol (ann.)", f"{ann_vol(spy_only):.2%}")
k3.metric("Mix Vol (ann.)", f"{ann_vol(mix):.2%}")
k4.metric("Mix Sharpe", f"{sharpe(mix, rf_annual):.2f}")

st.divider()

# -----------------------------
# Tabs layout (cleaner dashboard)
# -----------------------------
tab_perf, tab_risk, tab_rel, tab_table = st.tabs(["PERFORMANCE", "RISK", "RELATIONSHIP", "TABLE"])

with tab_perf:
    st.subheader("Cumulative Growth of $1")

    growth = pd.DataFrame(
        {
            "SPY Only": (1 + spy_only).cumprod(),
            mix_label: (1 + mix).cumprod(),
        },
        index=rets.index
    )
    growth.index.name = "timestamp"
    growth_long = to_long(growth, value_name="Growth")

    fig_growth = px.line(growth_long, x="Date", y="Growth", color="Series")
    fig_growth.update_layout(
        title=None,
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        legend_title_text="Series",
        hovermode="x unified"
    )
    st.plotly_chart(fig_growth, use_container_width=True)

with tab_risk:
    st.subheader("Rolling Annualized Volatility")

    rolling_vol = pd.DataFrame(
        {
            "SPY Only": spy_only.rolling(roll_window).std() * np.sqrt(252),
            mix_label: mix.rolling(roll_window).std() * np.sqrt(252),
        },
        index=rets.index
    ).dropna()

    if rolling_vol.empty:
        st.warning("Rolling window too large for the selected range. Reduce the window or expand the date range.")
        st.stop()

    rolling_vol.index.name = "timestamp"
    rolling_vol_long = to_long(rolling_vol, value_name="Annualized Vol")

    fig_vol = px.line(rolling_vol_long, x="Date", y="Annualized Vol", color="Series")
    fig_vol.update_layout(
        title=None,
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        legend_title_text="Series",
        hovermode="x unified"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with tab_rel:
    st.subheader("Rolling Correlation (BTC vs SPY)")

    rolling_corr = rets["BTC Close"].rolling(roll_window).corr(rets["SPY Close"]).dropna()
    if rolling_corr.empty:
        st.warning("Rolling correlation could not be computed for this window/range. Try a smaller window.")
        st.stop()

    corr_df = rolling_corr.to_frame("Rolling Corr").reset_index()
    corr_df.columns = ["Date", "Rolling Corr"]

    fig_corr = px.line(corr_df, x="Date", y="Rolling Corr")
    fig_corr.update_layout(
        title=None,
        xaxis_title="Date",
        yaxis_title="Rolling Correlation",
        hovermode="x unified"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_table:
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
        ),
        use_container_width=True
    )
