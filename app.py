# --- Ensure numeric dtypes (important for reliable pct_change/rolling) ---
df_f["SPY Close"] = pd.to_numeric(df_f["SPY Close"], errors="coerce")
df_f["BTC Close"] = pd.to_numeric(df_f["BTC Close"], errors="coerce")
df_f = df_f.dropna(subset=["SPY Close", "BTC Close"])

rets = compute_returns(df_f)

spy_only = port_rets(rets, 1.0, 0.0)
mix = port_rets(rets, w_spy, w_btc)

# --- Cumulative Growth of $1 (reset index for Plotly reliability) ---
growth = pd.DataFrame({
    "Date": spy_only.index,
    "SPY Only": (1 + spy_only).cumprod().values,
    f"{int(w_spy*100)}/{int(w_btc*100)} SPY/BTC": (1 + mix).cumprod().values,
})

fig_growth = px.line(
    growth,
    x="Date",
    y=[c for c in growth.columns if c != "Date"],
    title="Cumulative Growth of $1"
)
st.plotly_chart(fig_growth, use_container_width=True)

# --- Rolling Annualized Volatility (reset index + use roll_window slider) ---
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
