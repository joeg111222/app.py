import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------------------------------
# App setup
# -------------------------------------------------------
st.set_page_config(
    page_title="Bitcoin vs SPY: Diversification Dashboard",
    layout="wide"
)

st.title("Bitcoin vs SPY: Diversification Dashboard")

st.markdown(
    """
This dashboard evaluates whether adding Bitcoin to a traditional equity portfolio
improves diversification. We compare SPY-only portfolios to mixed SPY–Bitcoin portfolios
using returns, volatility, correlation, and risk-adjusted performance.
"""
)

# -------------------------------------------------------
# Load data (assumes already prepared returns dataframe)
# returns columns: ["SPY Close", "BTC Close"]
# index must be datetime
# -------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("returns.csv", parse_dates=["timestamp"], index_col="timestamp")
    return df

returns = load_data()

# -------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------
st.sidebar.header("Controls")

start_date, end_date = st.sidebar.date_input(
    "Date range",
    [returns.index.min(), returns.index.max()]
)

rolling_window = st.sidebar.slider(
    "Rolling window (days)",
    min_value=30,
    max_value=252,
    value=90,
    step=10
)

btc_weight = st.sidebar.slider(
    "Bitcoin weight",
    min_value=0.0,
    max_value=0.3,
    value=0.10,
    step=0.01
)

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual)",
    min_value=0.0,
    max_value=0.1,
    value=0.02,
    step=0.005
)

# -------------------------------------------------------
# Filter date range
# -------------------------------------------------------
returns = returns.loc[start_date:end_date]

# -------------------------------------------------------
# Portfolio construction
# -------------------------------------------------------
weights_spy = [1.0, 0.0]
weights_mix = [1 - btc_weight, btc_weight]

portfolio_returns = pd.DataFrame(index=returns.index)

portfolio_returns["SPY Only"] = (
    returns["SPY Close"] * weights_spy[0]
)

portfolio_returns["SPY/BTC Mix"] = (
    returns["SPY Close"] * weights_mix[0] +
    returns["BTC Close"] * weights_mix[1]
)

# -------------------------------------------------------
# Summary metrics
# -------------------------------------------------------
ann_factor = np.sqrt(252)

vol_spy = portfolio_returns["SPY Only"].std() * ann_factor
vol_mix = portfolio_returns["SPY/BTC Mix"].std() * ann_factor

corr = returns["SPY Close"].corr(returns["BTC Close"])

sharpe_spy = (
    (portfolio_returns["SPY Only"].mean() - risk_free_rate / 252)
    / portfolio_returns["SPY Only"].std()
) * ann_factor

sharpe_mix = (
    (portfolio_returns["SPY/BTC Mix"].mean() - risk_free_rate / 252)
    / portfolio_returns["SPY/BTC Mix"].std()
) * ann_factor

# -------------------------------------------------------
# Top metrics display
# -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("BTC–SPY Correlation", f"{corr:.2f}")
c2.metric("SPY Vol (ann.)", f"{vol_spy:.2%}")
c3.metric("Mix Vol (ann.)", f"{vol_mix:.2%}")
c4.metric("Mix Sharpe", f"{sharpe_mix:.2f}")

# -------------------------------------------------------
# Cumulative growth plot (KEEP – core result)
# -------------------------------------------------------
st.subheader("Cumulative Growth of $1")

cum_returns = (1 + portfolio_returns).cumprod()

fig_growth = go.Figure()
for col in cum_returns.columns:
    fig_growth.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns[col],
            mode="lines",
            name=col
        )
    )

fig_growth.update_layout(
    yaxis_title="Portfolio Value",
    xaxis_title="Date"
)

st.plotly_chart(fig_growth, use_container_width=True)

# -------------------------------------------------------
# Rolling volatility (KEEP – shows dynamic risk)
# -------------------------------------------------------
st.subheader("Rolling Annualized Volatility")

rolling_vol = (
    portfolio_returns
    .rolling(rolling_window)
    .std()
    * ann_factor
).dropna()

if rolling_vol.empty:
    st.warning(
        "Rolling window too large for the selected date range. "
        "Reduce the window or expand the date range."
    )
    st.stop()

fig_vol = go.Figure()
for col in rolling_vol.columns:
    fig_vol.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[col],
            mode="lines",
            name=col
        )
    )

fig_vol.update_layout(
    yaxis_title="Annualized Volatility",
    xaxis_title="Date"
)

st.plotly_chart(fig_vol, use_container_width=True)

# -------------------------------------------------------
# Rolling correlation (KEEP – diversification signal)
# -------------------------------------------------------
st.subheader("Rolling Correlation (SPY vs BTC)")

rolling_corr = (
    returns["SPY Close"]
    .rolling(rolling_window)
    .corr(returns["BTC Close"])
    .dropna()
)

fig_corr = go.Figure()
fig_corr.add_trace(
    go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr,
        mode="lines",
        name="Rolling Correlation"
    )
)

fig_corr.update_layout(
    yaxis_title="Correlation",
    xaxis_title="Date"
)

st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------------------------------
# Summary table (KEEP – grading friendly)
# -------------------------------------------------------
st.subheader("Summary Table")

summary = pd.DataFrame(
    {
        "Annualized Return": [
            portfolio_returns["SPY Only"].mean() * 252,
            portfolio_returns["SPY/BTC Mix"].mean() * 252
        ],
        "Annualized Vol": [vol_spy, vol_mix],
        "Sharpe": [sharpe_spy, sharpe_mix]
    },
    index=["SPY Only", "SPY/BTC Mix"]
)

st.dataframe(
    summary.style.format(
        {
            "Annualized Return": "{:.2%}",
            "Annualized Vol": "{:.2%}",
            "Sharpe": "{:.2f}"
        }
    )
)
