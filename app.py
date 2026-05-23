# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Financial Analysis Dashboard", layout="wide", page_icon="📈")

# Custom CSS for stock market style (without "Wall Street" text)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00b4d8;
    }
    .recommend-buy {
        color: #00ff9d;
        font-weight: bold;
    }
    .recommend-sell {
        color: #ff4d4d;
        font-weight: bold;
    }
    .recommend-hold {
        color: #ffcc00;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Helper functions
# -------------------------------

@st.cache_data(ttl=86400)
def fetch_data(tickers, start, end, interval="1d"):
    """Fetch adjusted close and OHLCV for given tickers."""
    data = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=False, threads=True)
    if len(tickers) == 1:
        data = {tickers[0]: data}
    else:
        data = {ticker: data[ticker] for ticker in tickers if ticker in data}
    return data

@st.cache_data(ttl=86400)
def fetch_benchmark(benchmark, start, end, interval="1d"):
    """Fetch benchmark data."""
    bench = yf.download(benchmark, start=start, end=end, interval=interval, auto_adjust=False)
    return bench['Adj Close'] if 'Adj Close' in bench else bench['Close']

def heikin_ashi(df):
    """Compute Heikin Ashi OHLC from standard OHLC."""
    ha = pd.DataFrame(index=df.index)
    ha['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    ha['HA_Open'].iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    ha['HA_High'] = ha[['HA_Open', 'HA_Close']].join(df['High']).max(axis=1)
    ha['HA_Low'] = ha[['HA_Open', 'HA_Close']].join(df['Low']).min(axis=1)
    return ha

def add_technical_indicators(df, ema_windows=[7,30,50,200], bb_period=20, bb_std=2):
    """Add EMA, Bollinger Bands, RSI, MACD."""
    df = df.copy()
    # EMA
    for w in ema_windows:
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(window=bb_period).mean()
    df['BB_upper'] = df['BB_mid'] + bb_std * df['Close'].rolling(window=bb_period).std()
    df['BB_lower'] = df['BB_mid'] - bb_std * df['Close'].rolling(window=bb_period).std()
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def compute_risk_metrics(returns, rf_rate=0.0457, annualization_factor=252):
    """Compute daily and annualized metrics."""
    if len(returns) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    daily_ret = returns.mean()
    daily_vol = returns.std()
    annual_ret = (1 + daily_ret) ** annualization_factor - 1
    annual_vol = daily_vol * np.sqrt(annualization_factor)
    sharpe = (annual_ret - rf_rate) / annual_vol if annual_vol != 0 else np.nan
    return daily_ret, daily_vol, annual_ret, annual_vol, sharpe

def calculate_beta(asset_returns, market_returns):
    """Calculate beta against market."""
    if len(asset_returns) < 2 or len(market_returns) < 2:
        return np.nan
    # Ensure 1D arrays
    asset_returns = np.asarray(asset_returns).flatten()
    market_returns = np.asarray(market_returns).flatten()
    covariance = np.cov(asset_returns, market_returns)[0][1]
    variance = np.var(market_returns)
    return covariance / variance if variance != 0 else np.nan

def calculate_var(returns, capital=10000, confidence=0.95, horizon=1):
    """Value at Risk (historical method)."""
    if len(returns) == 0:
        return np.nan, np.nan
    var_percent = np.percentile(returns, (1-confidence)*100)
    var_absolute = capital * var_percent * np.sqrt(horizon)
    return var_percent, var_absolute

def forecast_price(series, days=63):
    """Forecast next 'days' using linear regression on time index."""
    df = series.dropna().reset_index()
    df['days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['days']].values
    y = df[series.name].values
    if len(X) < 2:
        return pd.Series(dtype=float), None
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(df['days'].max()+1, df['days'].max()+days+1).reshape(-1,1)
    forecast = model.predict(future_days)
    last_date = df['Date'].max()
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    return pd.Series(forecast, index=forecast_dates, name='Forecast'), model

def generate_recommendation(price_series, indicators, fundamentals, forecast_signal):
    """Rule-based + ML forecast recommendation."""
    if indicators.empty:
        return "HOLD", "recommend-hold", 0
    latest = indicators.iloc[-1]
    # Trend signals
    ema_7 = latest.get('EMA_7', latest.get('Close', 0))
    ema_30 = latest.get('EMA_30', ema_7)
    ema_200 = latest.get('EMA_200', ema_30)
    close = latest['Close']
    # RSI
    rsi = latest.get('RSI', 50)
    # MACD
    macd = latest.get('MACD', 0)
    signal = latest.get('Signal', 0)
    # Fundamentals (simplified)
    pe = fundamentals.get('P/E', 20) if fundamentals else 20
    # Score
    score = 0
    if close > ema_7 > ema_30 > ema_200:
        score += 2
    elif close > ema_7 and ema_7 > ema_30:
        score += 1
    elif close < ema_7 < ema_30:
        score -= 1
    if rsi < 30:
        score += 2
    elif rsi > 70:
        score -= 2
    if macd > signal:
        score += 1
    elif macd < signal:
        score -= 1
    if forecast_signal == 1:
        score += 1
    elif forecast_signal == -1:
        score -= 1
    if pe < 15:
        score += 1
    elif pe > 25:
        score -= 1
    
    if score >= 2:
        rec = "BUY"
        color = "recommend-buy"
    elif score <= -2:
        rec = "SELL"
        color = "recommend-sell"
    else:
        rec = "HOLD"
        color = "recommend-hold"
    return rec, color, score

def fetch_fundamentals(ticker):
    """Get key fundamental metrics using yfinance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamentals = {
        'Market Cap': info.get('marketCap'),
        'P/E': info.get('trailingPE'),
        'P/B': info.get('priceToBook'),
        'EPS': info.get('trailingEps'),
        'Dividend Yield': info.get('dividendYield'),
        'Free Cash Flow': info.get('freeCashflow'),
        'Revenue': info.get('totalRevenue'),
        'Gross Profit': info.get('grossProfits'),
    }
    return fundamentals

def fetch_recent_news(ticker):
    """Fetch news headlines from yfinance."""
    stock = yf.Ticker(ticker)
    news = stock.news[:5] if hasattr(stock, 'news') else []
    headlines = [item['title'] for item in news] if news else ["No recent news available."]
    return headlines

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    show_sidebar = st.checkbox("Show/Hide Sidebar", value=True, help="Toggle to hide sidebar (refresh to show again)")
    if not show_sidebar:
        st.stop()
    
    ticker_input = st.text_input("Ticker Symbols (comma separated)", value="AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    benchmark = st.text_input("Benchmark Index (e.g., ^GSPC for S&P500)", value="^GSPC")
    
    start_date = st.date_input("Start Date", datetime(2020,1,1))
    end_date = st.date_input("End Date", datetime.today())
    freq = st.selectbox("Data Frequency", ["Daily", "Weekly", "Monthly"])
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    interval = interval_map[freq]
    
    price_type = st.selectbox("Price Type", ["Adj Close", "Close"])
    rf_rate = st.number_input("Risk-Free Rate (annual)", value=0.0457, format="%.4f")
    confidence_level = st.slider("Confidence Level for VaR", 0.80, 0.99, 0.95, 0.01)
    
    st.markdown("---")
    st.markdown("### Technical Parameters")
    ema_windows = st.multiselect("EMA Periods", [7,30,50,200], default=[7,30,50,200])
    bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
    bb_std = st.slider("Bollinger Bands Std Dev", 1.0, 3.0, 2.0, 0.5)
    
    forecast_days = st.number_input("Forecast Horizon (days)", min_value=5, max_value=180, value=63)

# -------------------------------
# Main Dashboard
# -------------------------------
st.title("📊 Financial Analysis Dashboard")
st.markdown("Real-time risk & return analytics, technical indicators, and ML forecasts.")

if len(tickers) == 0:
    st.warning("Please enter at least one ticker symbol.")
    st.stop()

# Fetch data
with st.spinner("Fetching market data..."):
    data_dict = fetch_data(tickers, start_date, end_date, interval)
    benchmark_data = fetch_benchmark(benchmark, start_date, end_date, interval)

# Check if data exists
valid_tickers = [t for t in tickers if t in data_dict and not data_dict[t].empty]
if not valid_tickers:
    st.error("No valid data found for the given tickers. Check symbols or date range.")
    st.stop()

# Prepare returns and metrics
all_returns = pd.DataFrame()
all_prices = pd.DataFrame()
all_volatility = {}
all_betas = {}
all_sharpe = {}
all_var = {}
technical_dict = {}
fundamental_dict = {}
recommendations = {}
forecast_series_dict = {}

for ticker in valid_tickers:
    df = data_dict[ticker]
    # Use either Adj Close or Close
    price_col = price_type if price_type in df.columns else 'Close'
    prices = df[price_col].copy()
    prices.name = ticker
    all_prices[ticker] = prices
    returns = prices.pct_change().dropna()
    all_returns[ticker] = returns
    
    # Risk metrics
    daily_ret, daily_vol, ann_ret, ann_vol, sharpe = compute_risk_metrics(returns, rf_rate)
    all_volatility[ticker] = ann_vol
    all_sharpe[ticker] = sharpe
    # Beta against benchmark
    if benchmark_data is not None and not benchmark_data.empty:
        bench_returns = benchmark_data.pct_change().dropna()
        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) > 1:
            beta = calculate_beta(returns.loc[common_idx].values, bench_returns.loc[common_idx].values)
            all_betas[ticker] = beta
        else:
            all_betas[ticker] = np.nan
    else:
        all_betas[ticker] = np.nan
    # VaR
    var_percent, var_abs = calculate_var(returns, capital=10000, confidence=confidence_level)
    all_var[ticker] = {'percent': var_percent, 'absolute': var_abs}
    
    # Technical indicators
    tech_df = add_technical_indicators(df[['Open','High','Low','Close']], ema_windows, bb_period, bb_std)
    technical_dict[ticker] = tech_df
    
    # Fundamentals
    try:
        fund = fetch_fundamentals(ticker)
        fundamental_dict[ticker] = fund
    except:
        fundamental_dict[ticker] = {}
    
    # Forecast
    forecast_series, model = forecast_price(prices.dropna(), days=forecast_days)
    forecast_series_dict[ticker] = forecast_series
    # Forecast direction
    if not forecast_series.empty and len(prices) > 0:
        last_price = prices.iloc[-1]
        forecast_end = forecast_series.iloc[-1]
        forecast_signal = 1 if forecast_end > last_price * 1.02 else (-1 if forecast_end < last_price * 0.98 else 0)
    else:
        forecast_signal = 0
    
    # Recommendation
    rec, rec_color, score = generate_recommendation(prices, tech_df, fundamental_dict[ticker], forecast_signal)
    recommendations[ticker] = {'rec': rec, 'color': rec_color, 'score': score, 'forecast_signal': forecast_signal}

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🎯 Benchmark", "🔥 Correlation", "📉 Covariance", "📥 Download"])

# -------------------------------
# Tab 1: Overview
# -------------------------------
with tab1:
    st.subheader("Asset Summary")
    summary_df = pd.DataFrame({
        'Ticker': valid_tickers,
        'Annualized Return (%)': [all_returns[t].mean() * 252 * 100 if not all_returns[t].empty else np.nan for t in valid_tickers],
        'Annualized Volatility (%)': [all_volatility[t]*100 if not np.isnan(all_volatility[t]) else np.nan for t in valid_tickers],
        'Sharpe Ratio': [all_sharpe[t] for t in valid_tickers],
        'Beta': [all_betas.get(t, np.nan) for t in valid_tickers],
        'VaR (95%, 1d %)': [all_var[t]['percent']*100 if not np.isnan(all_var[t]['percent']) else np.nan for t in valid_tickers],
        'Recommendation': [recommendations[t]['rec'] for t in valid_tickers]
    })
    st.dataframe(summary_df.style.format({
        'Annualized Return (%)': '{:.2f}',
        'Annualized Volatility (%)': '{:.2f}',
        'Sharpe Ratio': '{:.3f}',
        'Beta': '{:.2f}',
        'VaR (95%, 1d %)': '{:.2f}'
    }))
    
    # For each ticker, show charts and recommendations
    for ticker in valid_tickers:
        st.markdown(f"### {ticker} – {recommendations[ticker]['rec']}")
        col1, col2 = st.columns([3,1])
        with col1:
            # Heikin Ashi candlestick chart with EMA & BB
            df_ohlc = data_dict[ticker][['Open','High','Low','Close']].copy()
            ha = heikin_ashi(df_ohlc)
            tech = technical_dict[ticker].copy()
            # Plot using Plotly
            fig = go.Figure()
            # Heikin Ashi candlesticks
            fig.add_trace(go.Candlestick(
                x=ha.index,
                open=ha['HA_Open'], high=ha['HA_High'], low=ha['HA_Low'], close=ha['HA_Close'],
                name='Heikin Ashi', increasing_line_color='#00ff9d', decreasing_line_color='#ff4d4d'
            ))
            # EMAs
            for w in ema_windows:
                if f'EMA_{w}' in tech.columns:
                    fig.add_trace(go.Scatter(x=tech.index, y=tech[f'EMA_{w}'], mode='lines', name=f'EMA {w}', line=dict(width=1)))
            # Bollinger Bands
            if 'BB_upper' in tech.columns:
                fig.add_trace(go.Scatter(x=tech.index, y=tech['BB_upper'], mode='lines', name='BB Upper', line=dict(dash='dash', color='gray')))
                fig.add_trace(go.Scatter(x=tech.index, y=tech['BB_lower'], mode='lines', name='BB Lower', line=dict(dash='dash', color='gray')))
                fig.add_trace(go.Scatter(x=tech.index, y=tech['BB_mid'], mode='lines', name='BB Mid', line=dict(color='orange')))
            
            fig.update_layout(title=f'{ticker} – Heikin Ashi with Technicals', xaxis_title='Date', yaxis_title='Price', height=500, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast chart
            prices = all_prices[ticker].dropna()
            forecast = forecast_series_dict[ticker]
            if not forecast.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Historical Price'))
                fig2.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name=f'{forecast_days}-day Forecast', line=dict(dash='dot', color='cyan')))
                fig2.update_layout(title=f'{ticker} – Price Forecast (Linear Regression)', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Not enough data for forecast.")
        
        with col2:
            rec = recommendations[ticker]['rec']
            color = recommendations[ticker]['color']
            st.markdown(f"<div class='metric-card'><h4>Recommendation</h4><p class='{color}' style='font-size:24px'>{rec}</p>", unsafe_allow_html=True)
            st.markdown(f"**Score:** {recommendations[ticker]['score']}/?")
            st.markdown("**Forecast Signal:** " + ("Bullish" if recommendations[ticker]['forecast_signal']==1 else "Bearish" if recommendations[ticker]['forecast_signal']==-1 else "Neutral"))
            # Fundamentals
            fund = fundamental_dict[ticker]
            if fund:
                st.markdown("**Key Fundamentals**")
                st.write(f"P/E: {fund.get('P/E', 'N/A')}")
                st.write(f"EPS: {fund.get('EPS', 'N/A')}")
                st.write(f"FCF: {fund.get('Free Cash Flow', 'N/A')}")
            # News
            st.markdown("**Recent News**")
            news = fetch_recent_news(ticker)
            for n in news[:3]:
                st.write(f"- {n[:80]}...")
            
            # Insights & Risks
            st.markdown("**Insights & Risks**")
            if rec == "BUY":
                st.success("✅ Undervalued relative to peers; positive technical momentum.")
            elif rec == "SELL":
                st.error("⚠️ Overbought signals; weakening fundamentals.")
            else:
                st.info("⏸️ Neutral zone – watch for breakout or macro catalysts.")
            st.markdown("**Macro Considerations**")
            st.write("• Interest rate expectations\n• Sector rotation trends\n• Global liquidity conditions")
    
    # VIX indicator
    vix = yf.Ticker("^VIX")
    vix_hist = vix.history(start=start_date, end=end_date, interval=interval)
    if not vix_hist.empty:
        st.subheader("Market Fear Gauge: VIX")
        fig_vix = px.line(vix_hist, x=vix_hist.index, y='Close', title="VIX - Volatility Index", template='plotly_dark')
        st.plotly_chart(fig_vix, use_container_width=True)

# -------------------------------
# Tab 2: Benchmark Comparison
# -------------------------------
with tab2:
    if benchmark_data is not None and not benchmark_data.empty:
        # Cumulative returns
        cum_returns = (1 + all_returns).cumprod() - 1
        bench_ret = benchmark_data.pct_change().dropna()
        bench_cum = (1 + bench_ret).cumprod() - 1
        bench_cum.name = benchmark
        comp_df = pd.concat([cum_returns, bench_cum], axis=1).dropna()
        if not comp_df.empty:
            fig_bench = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title=f"Cumulative Returns vs {benchmark}", template='plotly_dark')
            st.plotly_chart(fig_bench, use_container_width=True)
        else:
            st.warning("Insufficient overlapping data for benchmark comparison.")
        
        # Relative performance table
        if not comp_df.empty:
            total_ret = comp_df.iloc[-1] * 100
            st.dataframe(pd.DataFrame(total_ret.sort_values(ascending=False), columns=['Total Return (%)']).style.format('{:.2f}'))
    else:
        st.warning("Benchmark data not available.")

# -------------------------------
# Tab 3: Correlation Heatmap
# -------------------------------
with tab3:
    if len(valid_tickers) > 1 and not all_returns.empty:
        corr_matrix = all_returns.corr()
        fig_corr = ff.create_annotated_heatmap(z=corr_matrix.values, x=list(corr_matrix.columns), y=list(corr_matrix.index), colorscale='RdBu', showscale=True)
        fig_corr.update_layout(title="Correlation Matrix of Daily Returns", template='plotly_dark', height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need at least two assets for correlation heatmap.")

# -------------------------------
# Tab 4: Covariance Heatmap (annualized)
# -------------------------------
with tab4:
    if len(valid_tickers) > 1 and not all_returns.empty:
        annual_cov = all_returns.cov() * 252
        fig_cov = ff.create_annotated_heatmap(z=annual_cov.values, x=list(annual_cov.columns), y=list(annual_cov.index), colorscale='Viridis', showscale=True)
        fig_cov.update_layout(title="Annualized Covariance Matrix", template='plotly_dark', height=600)
        st.plotly_chart(fig_cov, use_container_width=True)
    else:
        st.info("Need at least two assets for covariance heatmap.")

# -------------------------------
# Tab 5: Download Excel Report
# -------------------------------
with tab5:
    st.subheader("Download Risk & Return Report")
    # Build a comprehensive DataFrame
    risk_df = pd.DataFrame({
        'Ticker': valid_tickers,
        'Annualized Return (%)': [all_returns[t].mean() * 252 * 100 if not all_returns[t].empty else np.nan for t in valid_tickers],
        'Annualized Volatility (%)': [all_volatility[t]*100 if not np.isnan(all_volatility[t]) else np.nan for t in valid_tickers],
        'Sharpe Ratio': [all_sharpe[t] for t in valid_tickers],
        'Beta (vs benchmark)': [all_betas.get(t, np.nan) for t in valid_tickers],
        'VaR (95%, 1d %)': [all_var[t]['percent']*100 if not np.isnan(all_var[t]['percent']) else np.nan for t in valid_tickers],
        'VaR (Absolute, 1d)': [all_var[t]['absolute'] if not np.isnan(all_var[t]['absolute']) else np.nan for t in valid_tickers],
        'Recommendation': [recommendations[t]['rec'] for t in valid_tickers],
    })
    # Add forecast end price
    forecast_end_vals = [forecast_series_dict[t].iloc[-1] if not forecast_series_dict[t].empty else np.nan for t in valid_tickers]
    risk_df[f'Forecast Price ({forecast_days}d)'] = forecast_end_vals
    
    # Technical indicators last values
    for t in valid_tickers:
        tech_last = technical_dict[t].iloc[-1] if not technical_dict[t].empty else pd.Series()
        for col in ['RSI', 'MACD', 'Signal']:
            if col in tech_last:
                risk_df.loc[risk_df['Ticker']==t, col] = tech_last[col]
    
    output = pd.ExcelWriter("financial_report.xlsx", engine='openpyxl')
    risk_df.to_excel(output, sheet_name='RiskMetrics', index=False)
    
    # Add fundamentals
    fund_list = []
    for t in valid_tickers:
        row = {'Ticker': t}
        row.update(fundamental_dict[t])
        fund_list.append(row)
    fund_df = pd.DataFrame(fund_list)
    fund_df.to_excel(output, sheet_name='Fundamentals', index=False)
    
    # Add historical data for each ticker
    for t in valid_tickers:
        hist = data_dict[t][['Open','High','Low','Close','Volume']].copy()
        if not hist.empty:
            hist['Returns'] = all_returns[t]
            hist.to_excel(output, sheet_name=f'{t}_Historical')
    
    output.close()
    
    with open("financial_report.xlsx", "rb") as f:
        st.download_button("📥 Download Full Excel Report", data=f, file_name="financial_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
