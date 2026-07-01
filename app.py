import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Elite Quant | Kelly-Kurtosis Engine", layout="wide")

# --- SIDEBAR: INSTITUTIONAL PARAMETERS ---
st.sidebar.header("⚙️ Quantitative Parameters")
ticker = st.sidebar.text_input("Asset Ticker", value="^NSEI")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.divider()
st.sidebar.subheader("Alpha & Regime Sensors")
fast_ma = st.sidebar.slider("Fast Signal Window", 10, 100, 20)
slow_ma = st.sidebar.slider("Slow Signal Window", 100, 300, 100)
kelly_fraction = st.sidebar.radio("Kelly Sizing Aggression", ["Full Kelly (Max Growth, Max Risk)", "Half-Kelly (Institutional Standard)"])

st.sidebar.divider()
st.sidebar.subheader("Black Swan Override")
kurtosis_window = st.sidebar.slider("Kurtosis Lookback (Days)", 20, 100, 60)
kurtosis_threshold = st.sidebar.number_input("Crash Risk Limit (Excess Kurtosis)", value=3.0)

# --- MAIN HEADER ---
st.title("🦅 Elite Quant: Dynamic Kelly & Tail-Risk Engine")
st.write("Mathematically optimal position sizing governed by the Kelly Criterion, bounded by real-time higher-order moment (Kurtosis) crash detection.")

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
    
    # --- 1. BASE SIGNAL GENERATION ---
    df['Market_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Fast_MA'] = df['Close'].rolling(window=fast_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(window=slow_ma).mean()
    df['Direction'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0) # Long only base
    
    # --- 2. DYNAMIC KELLY CRITERION SIZING ---
    # Calculate rolling win probabilities and payoff ratios over a 60-day window
    roll_window = 60
    
    # Isolate gains and losses
    df['Gain'] = np.where(df['Market_Return'] > 0, df['Market_Return'], 0)
    df['Loss'] = np.where(df['Market_Return'] < 0, abs(df['Market_Return']), 0)
    
    # Rolling averages
    df['Avg_Gain'] = df['Gain'].rolling(window=roll_window).mean()
    df['Avg_Loss'] = df['Loss'].rolling(window=roll_window).mean()
    df['Win_Rate'] = (df['Gain'] > 0).rolling(window=roll_window).sum() / roll_window
    
    # Avoid division by zero
    df['Avg_Loss'] = df['Avg_Loss'].replace(0, np.nan) 
    df['Payoff_Ratio'] = df['Avg_Gain'] / df['Avg_Loss']
    
    # The Optimal Kelly Formula
    df['Kelly_Fraction'] = df['Win_Rate'] - ((1 - df['Win_Rate']) / df['Payoff_Ratio'])
    df['Kelly_Fraction'] = df['Kelly_Fraction'].fillna(0)
    
    # Institutional bounds: Can't short in this model, max leverage 2x
    df['Kelly_Fraction'] = np.clip(df['Kelly_Fraction'], 0.0, 2.0)
    
    if kelly_fraction == "Half-Kelly (Institutional Standard)":
        df['Kelly_Fraction'] = df['Kelly_Fraction'] / 2.0
        
    # --- 3. THE BLACK SWAN RADAR (KURTOSIS OVERRIDE) ---
    # Excess Kurtosis > 3 indicates "fat tails" (high probability of extreme events)
    df['Rolling_Kurtosis'] = df['Market_Return'].rolling(window=kurtosis_window).kurt()
    
    # If Kurtosis spikes above our threshold, we override Kelly and force position to 0 (cash)
    df['Tail_Risk_Warning'] = np.where(df['Rolling_Kurtosis'] > kurtosis_threshold, 1, 0)
    
    # Final Exposure: Base Direction * Kelly Size * (1 - Tail Risk Warning)
    df['Final_Exposure'] = df['Direction'] * df['Kelly_Fraction'] * (1 - df['Tail_Risk_Warning'])
    
    # --- 4. RETURN CALCULATIONS ---
    # Shift exposure by 1 to avoid look-ahead bias
    df['Position'] = df['Final_Exposure'].shift(1).fillna(0)
    
    # Factor in 10bps transaction cost for turnover
    df['Trades'] = df['Position'].diff().abs().fillna(0)
    tc_pct = 10 / 10000.0 
    
    df['Strategy_Return'] = (df['Position'] * df['Market_Return']) - (df['Trades'] * tc_pct)
    df['Cumulative_Market'] = np.exp(df['Market_Return'].cumsum()) - 1
    df['Cumulative_Strategy'] = np.exp(df['Strategy_Return'].cumsum()) - 1
    
    # --- UI DASHBOARD ---
    st.markdown("### 📊 Absolute Return Metrics")
    
    trading_days = 252
    strat_vol = df['Strategy_Return'].std() * np.sqrt(trading_days)
    ann_strat_ret = df['Strategy_Return'].mean() * trading_days
    sharpe = (ann_strat_ret - 0.07) / strat_vol if strat_vol != 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Compound Strategy Return", f"{df['Cumulative_Strategy'].iloc[-1]*100:.2f}%")
    col2.metric("Market Benchmark", f"{df['Cumulative_Market'].iloc[-1]*100:.2f}%")
    col3.metric("Kelly-Adjusted Sharpe", f"{sharpe:.2f}")
    col4.metric("Tail Risk Events Avoided", f"{df['Tail_Risk_Warning'].sum()}")
    
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Performance", "Black Swan Radar", "The Mathematics"])
    
    with tab1:
        st.subheader("Equity Curve (Kelly Sizing + Crash Avoidance)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy']*100, name='Elite Kelly Strategy', line=dict(color='#00ff00', width=2.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market']*100, name='Market Benchmark', line=dict(color='gray', width=1.5, dash='dot')))
        fig.update_layout(height=500, template="plotly_dark", hovermode="x unified", yaxis_title="Return (%)")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("🚨 Tail-Risk Kurtosis & Position Sizing")
        st.write("Top chart: The percentage of capital mathematically allocated by the Kelly Formula. Bottom chart: Real-time Excess Kurtosis. When the red line spikes, the system overrides Kelly and moves to cash.")
        
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        
        # Plot Kelly Fraction
        fig2.add_trace(go.Scatter(x=df.index, y=df['Final_Exposure'], name='Allocated Leverage', fill='tozeroy', line=dict(color='#00bfff', width=1)), row=1, col=1)
        
        # Plot Kurtosis
        fig2.add_trace(go.Scatter(x=df.index, y=df['Rolling_Kurtosis'], name='Rolling Excess Kurtosis', line=dict(color='#ff3333', width=2)), row=2, col=1)
        
        # Add Threshold Line
        fig2.add_hline(y=kurtosis_threshold, line_dash="dot", line_color="yellow", annotation_text="Crash Limit", row=2, col=1)
        
        fig2.update_layout(height=600, template="plotly_dark", hovermode="x unified")
        fig2.update_yaxes(title_text="Capital Deployed (x)", row=1, col=1)
        fig2.update_yaxes(title_text="Kurtosis (Fat Tails)", row=2, col=1)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 🧠 The Institutional Mathematics")
        st.write("This engine abandons static position sizing. It uses the **Kelly Criterion** to mathematically optimize the fraction of the portfolio ($f^*$) to bet, based on the rolling probability of a win ($p$) and the gain/loss payoff ratio ($b$).")
        st.latex(r"f^* = p - \frac{1 - p}{b}")
        st.write("However, the Kelly Criterion assumes a normal distribution of returns. Because financial markets exhibit 'fat tails' (extreme, unpredictable crashes), this engine calculates the 4th moment of the distribution—**Kurtosis**.")
        st.latex(r"Kurtosis = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^4}{\left(\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2\right)^2}")
        st.write("If Excess Kurtosis breaches our limit, the system recognizes a non-normal regime and overrides the Kelly formula, instantly de-risking the portfolio to 0% exposure to avoid the Black Swan.")

except Exception as e:
    st.error(f"System Error: {e}")
