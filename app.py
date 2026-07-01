import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Institutional Quant Engine", layout="wide")

# --- SIDEBAR: INSTITUTIONAL PARAMETERS ---
st.sidebar.header("⚙️ Quantitative Parameters")
ticker = st.sidebar.text_input("Asset Ticker (e.g., ^NSEI, Reliance.NS)", value="^NSEI")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.divider()
st.sidebar.subheader("Strategy Logic")
fast_ma = st.sidebar.slider("Fast Moving Average", 10, 100, 50)
slow_ma = st.sidebar.slider("Slow Moving Average", 100, 300, 200)
strategy_type = st.sidebar.radio("Regime Type", ["Long Only", "Long / Short"])

st.sidebar.divider()
st.sidebar.subheader("Real-World Friction")
tc_bps = st.sidebar.number_input("Transaction Costs (Basis Points)", min_value=0, max_value=100, value=10)

# --- MAIN HEADER ---
st.title("🏛️ Institutional Backtest & Risk Engine")
st.write("Trend-following framework factoring real-world slippage, downside deviation, and stochastic forward modeling.")

@st.cache_data
def load_data(t, start, end):
    stock = yf.Ticker(t)
    df = stock.history(start=start, end=end)
    return df['Close']

try:
    data = load_data(ticker, start_date, end_date)
    if data.empty:
        st.error("No data found. Check ticker symbol.")
        st.stop()
        
    df = pd.DataFrame({'Close': data})
    
    # --- CORE MATH ENGINE ---
    df['Fast_MA'] = df['Close'].rolling(window=fast_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(window=slow_ma).mean()
    
    # Signal Generation
    if strategy_type == "Long Only":
        df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
    else: # Long/Short
        df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, -1)
        
    df['Position'] = df['Signal'].shift(1)
    
    # Transaction Costs (Turnover)
    df['Trades'] = df['Position'].diff().abs().fillna(0)
    tc_pct = tc_bps / 10000.0
    
    # Returns Calculation
    df['Market_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # Net Strategy Return = (Position * Market Return) - (Trades * Transaction Cost)
    df['Strategy_Return'] = (df['Position'] * df['Market_Return']) - (df['Trades'] * tc_pct)
    
    df['Cumulative_Market'] = np.exp(df['Market_Return'].cumsum()) - 1
    df['Cumulative_Strategy'] = np.exp(df['Strategy_Return'].cumsum()) - 1
    
    # --- ADVANCED RISK METRICS ---
    trading_days = 252
    risk_free_rate = 0.07 
    
    # Returns & Vol
    ann_market_ret = df['Market_Return'].mean() * trading_days
    ann_strat_ret = df['Strategy_Return'].mean() * trading_days
    strat_vol = df['Strategy_Return'].std() * np.sqrt(trading_days)
    
    # Max Drawdown
    cum_ret = np.exp(df['Strategy_Return'].cumsum())
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1
    max_dd = drawdown.min()
    
    # Sortino Ratio (Downside deviation only)
    downside_returns = df['Strategy_Return'][df['Strategy_Return'] < 0]
    downside_vol = downside_returns.std() * np.sqrt(trading_days)
    sortino = (ann_strat_ret - risk_free_rate) / downside_vol if downside_vol != 0 else 0
    
    # Sharpe Ratio
    sharpe = (ann_strat_ret - risk_free_rate) / strat_vol if strat_vol != 0 else 0

    # --- UI DASHBOARD ---
    st.markdown("### 📊 Performance Tearsheet")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Strategy Return", f"{df['Cumulative_Strategy'].iloc[-1]*100:.2f}%")
    col2.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Sortino Ratio", f"{sortino:.2f}")
    
    st.divider()

    # --- TABS FOR INSTITUTIONAL DEPTH ---
    tab1, tab2, tab3 = st.tabs(["Historical Equity Curve", "Monte Carlo Risk Engine", "Under the Hood"])
    
    with tab1:
        st.subheader("Cumulative Returns (Net of Fees)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy']*100, name='Strategy (Net)', line=dict(color='#00ff00', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market']*100, name='Market Benchmark', line=dict(color='gray', width=1.5, dash='dot')))
        fig.update_layout(height=500, template="plotly_dark", hovermode="x unified", yaxis_title="Return (%)")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("🎲 Stochastic Forward Projection (1-Year)")
        st.write("Simulating 100 random future equity paths based on the strategy's historical mean and volatility to assess tail risk.")
        
        # Monte Carlo Simulation
        sim_days = 252
        sims = 100
        mu = df['Strategy_Return'].mean()
        sigma = df['Strategy_Return'].std()
        
        # Generate random walks
        np.random.seed(42)
        sim_returns = np.random.normal(mu, sigma, (sim_days, sims))
        sim_prices = np.exp(np.cumsum(sim_returns, axis=0))
        
        # Plotting the fan chart
        mc_fig = go.Figure()
        for i in range(sims):
            mc_fig.add_trace(go.Scatter(y=sim_prices[:, i], mode='lines', line=dict(color='rgba(0, 255, 0, 0.05)'), showlegend=False))
        
        # Add Mean path
        mean_path = np.mean(sim_prices, axis=1)
        mc_fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color='white', width=3), name='Expected Mean Path'))
        
        mc_fig.update_layout(height=500, template="plotly_dark", yaxis_title="Projected Capital Multiplier", xaxis_title="Trading Days into Future")
        st.plotly_chart(mc_fig, use_container_width=True)

    with tab3:
        st.markdown("### 🏛️ Quant Methodology")
        st.write("This engine processes standard momentum variables but filters them through an institutional lens.")
        st.markdown(f"**Friction Accounting:** Every time the strategy flips positions, it deducts `{tc_bps}` basis points from the gross return to account for broker commissions and bid/ask slippage.")
        st.markdown("**Risk Adjusted Returns:** Unlike retail tools, we calculate the Sortino Ratio to evaluate whether the volatility is 'good' (upside) or 'bad' (downside).")
        st.latex(r"Sortino = \frac{R_p - R_f}{\sigma_d}")
        
except Exception as e:
    st.error(f"System Error: {e}")
