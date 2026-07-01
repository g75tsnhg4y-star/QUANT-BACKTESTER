import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Alpha Engine | Vol-Scaled", layout="wide")

# --- SIDEBAR: INSTITUTIONAL PARAMETERS ---
st.sidebar.header("⚙️ Strategy Parameters")
ticker = st.sidebar.text_input("Asset Ticker (e.g., ^NSEI, RELIANCE.NS)", value="^NSEI")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.divider()
st.sidebar.subheader("Alpha Generation (Signal)")
fast_ma = st.sidebar.slider("Fast Moving Average", 10, 100, 50)
slow_ma = st.sidebar.slider("Slow Moving Average", 100, 300, 200)
strategy_type = st.sidebar.radio("Regime Type", ["Long Only", "Long / Short"])

st.sidebar.divider()
st.sidebar.subheader("Risk & Sizing Engine")
vol_target = st.sidebar.slider("Target Annual Volatility (%)", 5, 40, 15) / 100.0
max_leverage = st.sidebar.number_input("Max Allowed Leverage (x)", min_value=1.0, max_value=3.0, value=1.5)
tc_bps = st.sidebar.number_input("Transaction Costs (Bps)", min_value=0, max_value=100, value=10)

# --- MAIN HEADER ---
st.title("🏛️ Volatility-Scaled Quantitative Engine")
st.write("Dynamic position sizing using inverse-volatility targeting, paired with momentum factor signals.")

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
    
    # --- 1. SIGNAL GENERATION ---
    df['Fast_MA'] = df['Close'].rolling(window=fast_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(window=slow_ma).mean()
    
    if strategy_type == "Long Only":
        df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
    else:
        df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, -1)
        
    df['Direction'] = df['Signal'].shift(1)
    
    # --- 2. DYNAMIC VOLATILITY SCALING (THE SECRET SAUCE) ---
    df['Market_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # Calculate 20-day annualized rolling volatility
    df['Rolling_Vol'] = df['Market_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Position Size = Target Volatility / Realized Volatility (Capped at max_leverage)
    df['Target_Exposure'] = vol_target / df['Rolling_Vol']
    df['Target_Exposure'] = df['Target_Exposure'].replace([np.inf, -np.inf], 0).fillna(0)
    df['Exposure'] = np.minimum(df['Target_Exposure'], max_leverage)
    
    # Final Position = Direction * Exposure
    df['Position'] = df['Direction'] * df['Exposure']
    
    # --- 3. RETURN CALCULATIONS ---
    df['Trades'] = df['Position'].diff().abs().fillna(0)
    tc_pct = tc_bps / 10000.0
    
    # Unscaled Benchmark for comparison
    df['Unscaled_Strategy'] = (df['Direction'] * df['Market_Return'])
    
    # Vol-Scaled Strategy Return (Net of fees)
    df['Strategy_Return'] = (df['Position'] * df['Market_Return']) - (df['Trades'] * tc_pct)
    
    df['Cumulative_Market'] = np.exp(df['Market_Return'].cumsum()) - 1
    df['Cumulative_Strategy'] = np.exp(df['Strategy_Return'].cumsum()) - 1
    
    # Drawdown Math
    cum_ret = np.exp(df['Strategy_Return'].cumsum())
    running_max = cum_ret.cummax()
    df['Drawdown'] = (cum_ret / running_max) - 1
    max_dd = df['Drawdown'].min()
    
    # --- 4. RISK METRICS ---
    trading_days = 252
    risk_free_rate = 0.07 
    
    ann_strat_ret = df['Strategy_Return'].mean() * trading_days
    strat_vol = df['Strategy_Return'].std() * np.sqrt(trading_days)
    sharpe = (ann_strat_ret - risk_free_rate) / strat_vol if strat_vol != 0 else 0
    
    downside_returns = df['Strategy_Return'][df['Strategy_Return'] < 0]
    downside_vol = downside_returns.std() * np.sqrt(trading_days)
    sortino = (ann_strat_ret - risk_free_rate) / downside_vol if downside_vol != 0 else 0

    # --- UI DASHBOARD ---
    st.markdown("### 📊 Institutional Tearsheet")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Scaled Return", f"{df['Cumulative_Strategy'].iloc[-1]*100:.2f}%")
    col2.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Realized Volatility", f"{strat_vol*100:.2f}%")
    
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Underwater Drawdown", "Exposure Dynamics", "Methodology"])
    
    with tab1:
        st.subheader("Cumulative Returns (Vol-Scaled vs Market)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy']*100, name='Vol-Scaled Strategy', line=dict(color='#00ff00', width=2.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market']*100, name='Market Benchmark', line=dict(color='gray', width=1.5, dash='dot')))
        fig.update_layout(height=500, template="plotly_dark", hovermode="x unified", yaxis_title="Return (%)")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("🌊 Underwater Profile (Drawdown)")
        st.write("Visualizing the depth and duration of historical peak-to-trough losses.")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['Drawdown']*100, fill='tozeroy', name='Drawdown', line=dict(color='#ff3333', width=1), fillcolor='rgba(255, 51, 51, 0.2)'))
        fig2.update_layout(height=400, template="plotly_dark", hovermode="x unified", yaxis_title="Drawdown (%)")
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab3:
        st.subheader("⚖️ Dynamic Position Sizing")
        st.write("Notice how the strategy cuts exposure (leverage drops) during periods of high market volatility to protect capital.")
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=df.index, y=df['Rolling_Vol']*100, name='Market Volatility (%)', line=dict(color='gray', width=1)), secondary_y=False)
        fig3.add_trace(go.Scatter(x=df.index, y=df['Exposure'], name='Strategy Leverage (x)', line=dict(color='#00bfff', width=1.5)), secondary_y=True)
        fig3.update_layout(height=400, template="plotly_dark", hovermode="x unified")
        fig3.update_yaxes(title_text="Market Volatility (%)", secondary_y=False)
        fig3.update_yaxes(title_text="Leverage Applied (x)", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("### 🏛️ The Institutional Edge: Inverse Volatility Scaling")
        st.write("Retail backtests assume a constant 100% exposure to the asset. This engine utilizes **Target Volatility Sizing**, automatically deleveraging during market crashes and leveraging up during quiet bull markets.")
        st.write("The position size at any given time $t$ is calculated as:")
        st.latex(r"Exposure_t = min\left( \frac{\sigma_{target}}{\sigma_{rolling, 20d}}, MaxLeverage \right)")
        st.write("This ensures the portfolio maintains a constant risk profile, vastly improving the Sortino ratio and protecting against left-tail events.")
        
except Exception as e:
    st.error(f"System Error: {e}")
