import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="God-Mode Quant Engine", layout="wide")

# --- SIDEBAR: INSTITUTIONAL PARAMETERS ---
st.sidebar.header("⚙️ Quantitative Parameters")
ticker = st.sidebar.text_input("Asset Ticker", value="^NSEI")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.divider()
st.sidebar.subheader("1. Alpha Generation")
fast_ma = st.sidebar.slider("Fast Moving Average", 10, 100, 20)
slow_ma = st.sidebar.slider("Slow Moving Average", 100, 300, 100)

st.sidebar.divider()
st.sidebar.subheader("2. Position Sizing Engine")
sizing_model = st.sidebar.radio("Sizing Philosophy", ["Volatility Scaling (Risk Parity)", "Dynamic Kelly (Optimal Growth)"])

# Conditional sliders based on sizing choice
if sizing_model == "Volatility Scaling (Risk Parity)":
    vol_target = st.sidebar.slider("Target Annual Volatility (%)", 5, 40, 15) / 100.0
    max_leverage = st.sidebar.number_input("Max Leverage (x)", value=1.5)
else:
    kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.5, help="1.0 is Full Kelly, 0.5 is Half-Kelly")
    max_leverage = st.sidebar.number_input("Max Leverage (x)", value=2.0)

st.sidebar.divider()
st.sidebar.subheader("3. Black Swan Radar")
kurtosis_window = st.sidebar.slider("Kurtosis Lookback", 20, 100, 60)
kurtosis_threshold = st.sidebar.number_input("Crash Risk Limit", value=3.0)
tc_bps = st.sidebar.number_input("Transaction Costs (Bps)", value=10)

# --- MAIN HEADER ---
st.title("🏛️ Ultimate Modular Risk Engine")
st.write("A multi-regime quantitative backtester comparing Inverse-Volatility targeting vs Optimal Kelly sizing, protected by a 4th-moment Kurtosis crash override.")

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
    df['Market_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # --- 1. BASE SIGNAL ---
    df['Fast_MA'] = df['Close'].rolling(window=fast_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(window=slow_ma).mean()
    df['Direction'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
    
    # --- 2. SIZING ALGORITHMS ---
    # A. Volatility Scaling
    df['Rolling_Vol'] = df['Market_Return'].rolling(window=20).std() * np.sqrt(252)
    df['Vol_Exposure'] = vol_target / df['Rolling_Vol']
    df['Vol_Exposure'] = np.clip(df['Vol_Exposure'].replace([np.inf, -np.inf], 0).fillna(0), 0, max_leverage)
    
    # B. Kelly Criterion
    roll_window = 60
    df['Gain'] = np.where(df['Market_Return'] > 0, df['Market_Return'], 0)
    df['Loss'] = np.where(df['Market_Return'] < 0, abs(df['Market_Return']), 0)
    df['Avg_Gain'] = df['Gain'].rolling(window=roll_window).mean()
    df['Avg_Loss'] = df['Loss'].rolling(window=roll_window).mean().replace(0, np.nan)
    df['Win_Rate'] = (df['Gain'] > 0).rolling(window=roll_window).sum() / roll_window
    df['Payoff_Ratio'] = df['Avg_Gain'] / df['Avg_Loss']
    
    raw_kelly = df['Win_Rate'] - ((1 - df['Win_Rate']) / df['Payoff_Ratio'])
    if sizing_model == "Dynamic Kelly (Optimal Growth)":
        df['Kelly_Exposure'] = np.clip(raw_kelly.fillna(0) * kelly_fraction, 0, max_leverage)
    else:
        df['Kelly_Exposure'] = 0 # Ignored if Vol Scaling is selected
        
    # Select Active Sizing Model
    df['Raw_Exposure'] = df['Vol_Exposure'] if sizing_model == "Volatility Scaling (Risk Parity)" else df['Kelly_Exposure']
    
    # --- 3. BLACK SWAN OVERRIDE ---
    df['Rolling_Kurtosis'] = df['Market_Return'].rolling(window=kurtosis_window).kurt()
    df['Tail_Risk_Warning'] = np.where(df['Rolling_Kurtosis'] > kurtosis_threshold, 1, 0)
    
    # Final Position = Signal * Math Size * (1 - Crash Warning)
    df['Final_Position'] = df['Direction'] * df['Raw_Exposure'] * (1 - df['Tail_Risk_Warning'])
    
    # --- 4. RETURN MATH ---
    df['Position'] = df['Final_Position'].shift(1).fillna(0)
    df['Trades'] = df['Position'].diff().abs().fillna(0)
    tc_pct = tc_bps / 10000.0 
    
    df['Strategy_Return'] = (df['Position'] * df['Market_Return']) - (df['Trades'] * tc_pct)
    df['Cumulative_Market'] = np.exp(df['Market_Return'].cumsum()) - 1
    df['Cumulative_Strategy'] = np.exp(df['Strategy_Return'].cumsum()) - 1
    
    # Drawdown
    cum_ret = np.exp(df['Strategy_Return'].cumsum())
    running_max = cum_ret.cummax()
    df['Drawdown'] = (cum_ret / running_max) - 1
    max_dd = df['Drawdown'].min()
    
    # Metrics
    trading_days = 252
    strat_vol = df['Strategy_Return'].std() * np.sqrt(trading_days)
    ann_strat_ret = df['Strategy_Return'].mean() * trading_days
    sharpe = (ann_strat_ret - 0.07) / strat_vol if strat_vol != 0 else 0
    
    # --- UI DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy Return (Net)", f"{df['Cumulative_Strategy'].iloc[-1]*100:.2f}%")
    col2.metric("Market Benchmark", f"{df['Cumulative_Market'].iloc[-1]*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Performance Curve", "Sizing Dynamics", "Black Swan Radar", "Underwater Profile"])
    
    with tab1:
        st.subheader(f"Equity Curve ({sizing_model})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy']*100, name='Active Strategy', line=dict(color='#00ff00', width=2.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market']*100, name='Market Benchmark', line=dict(color='gray', width=1.5, dash='dot')))
        fig.update_layout(height=500, template="plotly_dark", hovermode="x unified", yaxis_title="Return (%)")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("⚖️ Position Sizing Engine")
        st.write("Observe how the selected mathematical model dynamically alters leverage prior to the Black Swan override.")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        if sizing_model == "Volatility Scaling (Risk Parity)":
            fig2.add_trace(go.Scatter(x=df.index, y=df['Rolling_Vol']*100, name='Market Volatility (%)', line=dict(color='gray', width=1)), secondary_y=False)
        else:
            fig2.add_trace(go.Scatter(x=df.index, y=df['Win_Rate']*100, name='Rolling Win Rate (%)', line=dict(color='gray', width=1)), secondary_y=False)
            
        fig2.add_trace(go.Scatter(x=df.index, y=df['Raw_Exposure'], name='Calculated Leverage (x)', line=dict(color='#00bfff', width=1.5)), secondary_y=True)
        fig2.update_layout(height=400, template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("🚨 Tail-Risk Kurtosis Override")
        st.write("When the red line (Kurtosis) spikes above the crash limit, the system overrides all sizing math and drops capital deployed to zero.")
        fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig3.add_trace(go.Scatter(x=df.index, y=df['Final_Position'], name='Actual Capital Deployed', fill='tozeroy', line=dict(color='#00bfff', width=1)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df.index, y=df['Rolling_Kurtosis'], name='Excess Kurtosis', line=dict(color='#ff3333', width=2)), row=2, col=1)
        fig3.add_hline(y=kurtosis_threshold, line_dash="dot", line_color="yellow", annotation_text="Crash Limit", row=2, col=1)
        fig3.update_layout(height=500, template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("🌊 Drawdown Profile")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.index, y=df['Drawdown']*100, fill='tozeroy', name='Drawdown', line=dict(color='#ff3333', width=1), fillcolor='rgba(255, 51, 51, 0.2)'))
        fig4.update_layout(height=350, template="plotly_dark", hovermode="x unified", yaxis_title="Drawdown (%)")
        st.plotly_chart(fig4, use_container_width=True)

except Exception as e:
    st.error(f"System Error: {e}")
