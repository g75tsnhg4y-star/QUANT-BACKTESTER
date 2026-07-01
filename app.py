import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Quant Backtester", layout="wide")

# --- SIDEBAR & INPUTS ---
st.sidebar.header("⚙️ Strategy Parameters")
ticker = st.sidebar.text_input("Index/Stock Ticker", value="^NSEI") # ^NSEI is Nifty 50
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
fast_ma = st.sidebar.slider("Fast Moving Average", 10, 100, 50)
slow_ma = st.sidebar.slider("Slow Moving Average", 100, 300, 200)

st.title("📈 Quantitative Momentum Backtester")
st.write("Testing a simple Moving Average Crossover strategy against Buy & Hold.")

# --- DATA PULL & BACKTEST LOGIC ---
@st.cache_data
def load_data(t, start, end):
    df = yf.download(t, start=start, end=end)
    return df['Close']

try:
    # 1. Fetch Data
    data = load_data(ticker, start_date, end_date)
    df = pd.DataFrame({'Close': data})
    
    # 2. Calculate Indicators
    df['Fast_MA'] = df['Close'].rolling(window=fast_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(window=slow_ma).mean()
    
    # 3. Generate Signals (1 = Buy, 0 = Sell/Cash)
    df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
    df['Position'] = df['Signal'].shift(1) # Shift by 1 to avoid look-ahead bias
    
    # 4. Calculate Returns
    df['Market_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Strategy_Return'] = df['Position'] * df['Market_Return']
    
    # Cumulative Returns
    df['Cumulative_Market'] = np.exp(df['Market_Return'].cumsum()) - 1
    df['Cumulative_Strategy'] = np.exp(df['Strategy_Return'].cumsum()) - 1
    
    # --- CALCULATE RISK METRICS ---
    trading_days = 252
    risk_free_rate = 0.07 # Assuming 7% for India
    
    # Annualized Returns
    ann_market_ret = (df['Market_Return'].mean() * trading_days)
    ann_strat_ret = (df['Strategy_Return'].mean() * trading_days)
    
    # Volatility
    strat_vol = df['Strategy_Return'].std() * np.sqrt(trading_days)
    
    # Sharpe Ratio
    sharpe_ratio = (ann_strat_ret - risk_free_rate) / strat_vol if strat_vol != 0 else 0

    # --- UI DASHBOARD ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy Total Return", f"{df['Cumulative_Strategy'].iloc[-1]*100:.2f}%")
    col2.metric("Market Buy & Hold Return", f"{df['Cumulative_Market'].iloc[-1]*100:.2f}%")
    col3.metric("Strategy Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # --- PLOTTING ---
    st.subheader("Performance vs Benchmark")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy']*100, name='Strategy Return (%)', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market']*100, name='Market Return (%)', line=dict(color='gray', width=1.5, dash='dot')))
    fig.update_layout(height=500, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Note on methodology
    st.markdown("### 📘 Methodology")
    st.write("The Sharpe Ratio is calculated as:")
    st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
    st.write("Where $R_p$ is the portfolio return, $R_f$ is the risk-free rate (7% assumed), and $\sigma_p$ is the annualized volatility of the strategy.")

except Exception as e:
    st.error("Failed to load data. Please check the ticker symbol.")