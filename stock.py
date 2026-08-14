from datetime import date, datetime, timedelta
import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings('ignore')

# Streamlit App Configuration
st.set_page_config(page_title="📈 Advanced Stock Analyzer Pro", layout="wide")

# API Keys (Loaded from secrets.toml safely)
NEWSAPI_KEY = st.secrets.get("newsapi", {}).get("api_key") if hasattr(st, "secrets") else None
FINNHUB_KEY = st.secrets.get("finnhub", {}).get("api_key") if hasattr(st, "secrets") else None

# Professional CSS Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .section-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">📈 Advanced Stock Analyzer Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        AI-Powered Technical Analysis, Sentiment Insights & Portfolio Optimization
    </p>
</div>
""", unsafe_allow_html=True)

# Predefined Stock List (~40 Popular Stocks)
POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "JNJ", "WMT", "MA", "UNH", "XOM", "PG", "HD", "CVX", "MRK", "ABBV",
    "PEP", "KO", "BAC", "COST", "TMO", "MCD", "CSCO", "ACN", "ABT", "LIN",
    "PFE", "DHR", "NFLX", "AMD", "INTC", "QCOM", "IBM", "DIS", "CAT", "GE"
]

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 📊 Stock Selection")
    
    # Dropdown for stock selection
    selected_stock_dropdown = st.selectbox(
        "Choose from popular stocks:",
        ["-- Select from list --"] + POPULAR_STOCKS
    )
    
    # Original text box where user can enter or modify the stock symbol
    default_text_val = selected_stock_dropdown if selected_stock_dropdown != "-- Select from list --" else "AAPL"
    stock_symbol = st.text_input("Or type Stock Symbol:", default_text_val).upper().strip()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date:", date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date:", date.today())
    
    st.markdown("---")
    st.markdown("#### 🔧 Analysis Settings")
    
    analysis_mode = st.selectbox(
        "Analysis Depth:",
        ["📊 Quick Overview", "📈 Technical Deep Dive", "🎯 Advanced AI Analytics"],
        index=1
    )
    
    include_news = st.checkbox("📰 Include News Sentiment", value=True)
    show_patterns = st.checkbox("🔍 Technical Pattern Recognition", value=True)
    risk_analysis = st.checkbox("⚠️ Risk Assessment", value=True)
    
    st.markdown("---")
    
    # API Status Check
    st.markdown("#### 🔌 API Status")
    st.caption(f"NewsAPI: {'✅ Connected' if NEWSAPI_KEY else '⚠️ No Key'}")
    st.caption(f"Finnhub: {'✅ Connected' if FINNHUB_KEY else '⚠️ No Key'}")
    
    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    with col_btn2:
        refresh_clicked = st.button("🔄 Refresh", use_container_width=True)

# Initialize Session State
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = None
if 'cached_info' not in st.session_state:
    st.session_state.cached_info = None
if 'cached_symbol' not in st.session_state:
    st.session_state.cached_symbol = ""

if refresh_clicked:
    st.cache_data.clear()
    st.session_state.run_analysis = False
    st.session_state.cached_data = None
    st.success("Cache cleared! Click 'Run Analysis' to reload fresh data.")

if run_clicked:
    if not stock_symbol:
        st.error("❌ Please enter a valid stock symbol.")
        st.session_state.run_analysis = False
    elif start_date >= end_date:
        st.error("❌ Invalid date range. Start date must be before end date.")
        st.session_state.run_analysis = False
    else:
        st.session_state.run_analysis = True
        st.session_state.cached_symbol = stock_symbol

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(symbol: str, start: date, end: date) -> tuple:
    """Fetch stock historical data and company info with caching and robust error handling."""
    progress_bar = st.progress(0, text="Initializing data download...")
    
    try:
        progress_bar.progress(25, text=f"Fetching company info for {symbol}...")
        stock = yf.Ticker(symbol)
        info = stock.info
        
        progress_bar.progress(50, text="Downloading historical price data...")
        data = yf.download(symbol, start=start, end=end, progress=False)
        
        if data.empty:
            progress_bar.empty()
            return None, None
        
        progress_bar.progress(75, text="Processing indicators and cleaning data...")
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in data.columns]
        
        # Standardize required columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in data.columns:
                matches = [c for c in data.columns if col.lower() in c.lower()]
                if matches:
                    data = data.rename(columns={matches[0]: col})
                else:
                    data[col] = np.nan
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.sort_index().ffill().bfill()
        
        # Technical Indicator Calculations
        close_series = data['Close']
        data['SMA_50'] = close_series.rolling(window=50, min_periods=1).mean()
        data['SMA_200'] = close_series.rolling(window=200, min_periods=1).mean()
        data['EMA_20'] = close_series.ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = close_series.ewm(span=12, adjust=False).mean()
        exp2 = close_series.ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        # Bollinger Bands
        data['BB_middle'] = close_series.rolling(window=20, min_periods=1).mean()
        bb_std = close_series.rolling(window=20, min_periods=1).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        
        # Volatility & Returns
        data['daily_returns'] = close_series.pct_change()
        data['volatility'] = data['daily_returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
        
        progress_bar.progress(100, text="Analysis complete!")
        progress_bar.empty()
        
        return data, info
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error fetching stock data for {symbol}: {str(e)}")
        return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_sentiment(symbol: str) -> tuple:
    """Fetch and analyze news sentiment with caching."""
    news_data = []
    
    if NEWSAPI_KEY:
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWSAPI_KEY}&language=en&sortBy=publishedAt&pageSize=10"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                for article in response.json().get('articles', []):
                    news_data.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', '#'),
                        'published': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', 'Unknown')
                    })
        except Exception:
            pass
            
    if not news_data and FINNHUB_KEY:
        try:
            from_date = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = date.today().strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                for article in response.json()[:10]:
                    news_data.append({
                        'title': article.get('headline', ''),
                        'description': article.get('summary', ''),
                        'url': article.get('url', '#'),
                        'published': datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d'),
                        'source': article.get('source', 'Finnhub')
                    })
        except Exception:
            pass

    if news_data:
        positive_keywords = ['up', 'bullish', 'growth', 'profit', 'gain', 'positive', 'buy', 'strong']
        negative_keywords = ['down', 'bearish', 'loss', 'decline', 'negative', 'sell', 'weak', 'risk']
        sentiments = []
        
        for news in news_data:
            text = f"{news['title']} {news['description']}".lower()
            pos = sum(1 for w in positive_keywords if w in text)
            neg = sum(1 for w in negative_keywords if w in text)
            sentiment = 1 if pos > neg else (-1 if neg > pos else 0)
            news['sentiment'] = sentiment
            sentiments.append(sentiment)
            
        avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0
        return news_data, avg_sentiment
    return [], 0.0

def detect_technical_patterns(data: pd.DataFrame) -> list:
    """Identify technical chart patterns."""
    patterns = []
    if len(data) < 50:
        return patterns
    
    close_prices = data['Close'].values
    high_prices = data['High'].values
    low_prices = data['Low'].values
    
    for i in range(50, len(close_prices) - 10):
        recent_highs = high_prices[i-20:i]
        recent_lows = low_prices[i-20:i]
        
        if len(recent_highs) >= 10 and high_prices[i] >= np.max(recent_highs[-10:]) * 0.98:
            patterns.append({'pattern': 'Double Top', 'date': data.index[i], 'price': high_prices[i], 'confidence': 'Medium'})
        if len(recent_lows) >= 10 and low_prices[i] <= np.min(recent_lows[-10:]) * 1.02:
            patterns.append({'pattern': 'Double Bottom', 'date': data.index[i], 'price': low_prices[i], 'confidence': 'Medium'})
            
    sma_50 = data['SMA_50'].values
    sma_200 = data['SMA_200'].values
    if len(sma_50) > 5 and len(sma_200) > 5:
        if sma_50[-1] > sma_200[-1] and sma_50[-5] <= sma_200[-5]:
            patterns.append({'pattern': 'Golden Cross', 'date': data.index[-1], 'price': close_prices[-1], 'confidence': 'High'})
        elif sma_50[-1] < sma_200[-1] and sma_50[-5] >= sma_200[-5]:
            patterns.append({'pattern': 'Death Cross', 'date': data.index[-1], 'price': close_prices[-1], 'confidence': 'High'})
            
    return patterns[-5:]

def calculate_risk_metrics(data: pd.DataFrame) -> dict:
    """Calculate risk metrics."""
    if len(data) < 30:
        return {}
    returns = data['daily_returns'].dropna()
    return {
        'annual_volatility': float(returns.std() * np.sqrt(252) * 100),
        'max_drawdown': float((data['Close'] / data['Close'].cummax() - 1).min() * 100),
        'sharpe_ratio': float((returns.mean() * 252) / (returns.std() * np.sqrt(252))) if returns.std() > 0 else 0.0,
        'var_95': float(np.percentile(returns, 5) * 100),
        'beta': 1.0
    }

def forecast_volatility(data: pd.DataFrame) -> tuple:
    """Simple volatility forecast trend."""
    if len(data) < 60:
        return None, None
    returns = data['daily_returns'].dropna().values[-60:]
    forecast_vol = float(np.mean(returns ** 2) * np.sqrt(252) * 100)
    recent_vol = data['volatility'].dropna().values[-20:]
    vol_trend = 'increasing' if recent_vol[-1] > recent_vol[0] else 'decreasing'
    return forecast_vol, vol_trend

def get_portfolio_suggestions(data: pd.DataFrame) -> list:
    """Generate portfolio action suggestions."""
    suggestions = []
    if len(data) < 50:
        return suggestions
    current_price = float(data['Close'].iloc[-1])
    sma_50 = float(data['SMA_50'].iloc[-1])
    sma_200 = float(data['SMA_200'].iloc[-1])
    rsi = float(data['RSI'].iloc[-1])
    
    if current_price > sma_50 > sma_200 and rsi < 70:
        suggestions.append({'action': 'BUY', 'reason': 'Strong uptrend with safe RSI', 'confidence': 'High', 'suggested_allocation': '3-5%'})
    elif current_price < sma_200 and rsi < 30:
        suggestions.append({'action': 'BUY', 'reason': 'Oversold condition near long-term support', 'confidence': 'Medium', 'suggested_allocation': '2-3%'})
    elif current_price > sma_50 and rsi > 70:
        suggestions.append({'action': 'SELL', 'reason': 'Overbought conditions present', 'confidence': 'Medium', 'suggested_allocation': 'Reduce position by 50%'})
    elif current_price < sma_50 < sma_200:
        suggestions.append({'action': 'SELL', 'reason': 'Established strong downtrend', 'confidence': 'High', 'suggested_allocation': 'Exit position'})
    else:
        suggestions.append({'action': 'HOLD', 'reason': 'Mixed technical indicators', 'confidence': 'Low', 'suggested_allocation': 'Maintain current position'})
    return suggestions

# Main Application Execution Logic
if st.session_state.run_analysis:
    # Use cached data if symbol matches to avoid re-fetching on tab clicks
    if st.session_state.cached_data is None or st.session_state.cached_symbol != stock_symbol:
        data, info = fetch_stock_data(stock_symbol, start_date, end_date)
        st.session_state.cached_data = data
        st.session_state.cached_info = info
        st.session_state.cached_symbol = stock_symbol
    else:
        data = st.session_state.cached_data
        info = st.session_state.cached_info

    if data is None or data.empty:
        st.error(f"⚠️ Could not retrieve data for ticker symbol '{stock_symbol}'. Please verify the ticker or date range.")
    else:
        st.markdown(f'<div class="section-header"><h3 style="margin: 0;">📊 Stock Overview: {stock_symbol}</h3></div>', unsafe_allow_html=True)
        st.caption(f"Last updated data timestamp: {data.index[-1].strftime('%Y-%m-%d')} | Records loaded: {len(data)}")
        
        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}%")
        with col2:
            market_cap = info.get('marketCap', 0) / 1e9 if info and 'marketCap' in info else 0
            st.metric("Market Cap", f"${market_cap:.1f}B" if market_cap > 0 else "N/A")
        with col3:
            volume = float(data['Volume'].iloc[-1])
            avg_volume = float(data['Volume'].rolling(20).mean().iloc[-1])
            st.metric("Volume", f"{volume:,.0f}", f"{(volume/avg_volume - 1)*100:+.1f}% vs avg")
        with col4:
            rsi_value = float(data['RSI'].iloc[-1])
            rsi_status = "Overbought" if rsi_value > 70 else ("Oversold" if rsi_value < 30 else "Neutral")
            st.metric("RSI (14)", f"{rsi_value:.1f}", rsi_status)

        # Tabs Setup
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Price Analysis", 
            "🤖 AI Insights", 
            "📰 News & Sentiment", 
            "⚖️ Risk & Portfolio"
        ])
        
        with tab1:
            col_chart, col_sidebar = st.columns([3, 1])
            with col_chart:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                    name="Price", increasing_line_color='#10b981', decreasing_line_color='#ef4444'
                ))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode="lines", name="50-Day MA", line=dict(color='#f59e0b', width=2)))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode="lines", name="200-Day MA", line=dict(color='#8b5cf6', width=2)))
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], mode="lines", name="BB Upper", line=dict(color='rgba(59, 130, 246, 0.3)', width=1), showlegend=False))
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], mode="lines", name="BB Lower", line=dict(color='rgba(59, 130, 246, 0.3)', width=1), fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)', showlegend=False))
                
                fig.update_layout(
                    title=f"{stock_symbol} Technical Price Chart",
                    xaxis_title="Date", yaxis_title="Price (USD)",
                    xaxis_rangeslider_visible=False, height=500, template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_sidebar:
                st.markdown("#### Key Indicators")
                # Fixed the Plotly Indicator title syntax error (changed set brackets to dictionary braces)
                rsi_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=rsi_value,
                    domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "RSI Indicator"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [{'range': [0, 30], 'color': "green"}, {'range': [30, 70], 'color': "lightgray"}, {'range': [70, 100], 'color': "red"}]
                    }
                ))
                rsi_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(rsi_gauge, use_container_width=True)
                
                macd_val = float(data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1])
                st.metric("MACD Diff", f"{macd_val:.4f}", "Bullish" if macd_val > 0 else "Bearish")
                volatility_val = float(data['volatility'].iloc[-1] * 100)
                st.metric("20-Day Volatility", f"{volatility_val:.1f}%")

        with tab2:
            st.markdown("#### 🤖 AI & Technical Pattern Insights")
            col_ai1, col_ai2 = st.columns(2)
            
            with col_ai1:
                if show_patterns:
                    st.markdown("##### 🔍 Detected Chart Patterns")
                    patterns = detect_technical_patterns(data)
                    if patterns:
                        for p in patterns:
                            st.info(f"**{p['pattern']}** detected on {p['date'].strftime('%Y-%m-%d')} at price ${p['price']:.2f} (Confidence: {p['confidence']})")
                    else:
                        st.info("No significant technical chart patterns found in the recent window.")
                        
                st.markdown("##### 📊 Volatility Forecast")
                f_vol, v_trend = forecast_volatility(data)
                if f_vol:
                    st.metric("Estimated Volatility Forecast", f"{f_vol:.1f}%", f"{v_trend} trend")
                    
            with col_ai2:
                st.markdown("##### 🎯 Portfolio Action Recommendations")
                suggestions = get_portfolio_suggestions(data)
                for s in suggestions:
                    st.markdown(f"**Action:** `{s['action']}`")
                    st.caption(f"Reason: {s['reason']}")
                    st.progress(0.8 if s['confidence'] == 'High' else 0.5)
                    st.caption(f"Suggested Allocation: {s['suggested_allocation']}")
                    st.divider()

        with tab3:
            if include_news:
                st.markdown("#### 📰 Market News & Sentiment Intelligence")
                news_articles, avg_sent = get_news_sentiment(stock_symbol)
                if news_articles:
                    col_n1, col_n2, col_n3 = st.columns(3)
                    col_n1.metric("Sentiment Score", f"{avg_sent * 100:.0f}%", "Positive" if avg_sent > 0 else "Negative" if avg_sent < 0 else "Neutral")
                    col_n2.metric("Positive Articles", sum(1 for n in news_articles if n['sentiment'] > 0))
                    col_n3.metric("Negative Articles", sum(1 for n in news_articles if n['sentiment'] < 0))
                    
                    st.markdown("---")
                    for art in news_articles[:5]:
                        icon = "🟢" if art['sentiment'] > 0 else ("🔴" if art['sentiment'] < 0 else "⚪")
                        st.markdown(f"**{icon} {art['title']}**")
                        if art['description']:
                            st.caption(art['description'])
                        st.caption(f"Source: {art['source']} | Published: {art['published'][:10]}")
                        if art['url'] != '#':
                            st.link_button("Read Article", art['url'])
                        st.divider()
                else:
                    st.info("No recent news found for this symbol. Verify API keys in secrets or try another stock.")
            else:
                st.info("News sentiment analysis is disabled in sidebar settings.")

        with tab4:
            if risk_analysis:
                st.markdown("#### ⚖️ Risk Metrics & Portfolio Exposure")
                metrics = calculate_risk_metrics(data)
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("Annualized Volatility", f"{metrics.get('annual_volatility', 0):.1f}%")
                    st.metric("Maximum Drawdown", f"{metrics.get('max_drawdown', 0):.1f}%")
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("Daily VaR (95%)", f"{metrics.get('var_95', 0):.1f}%")
                with col_r2:
                    drawdown_series = (data['Close'] / data['Close'].cummax() - 1) * 100
                    fig_dd = px.area(x=data.index, y=drawdown_series, title="Historical Drawdown (%)", labels={'x': 'Date', 'y': 'Drawdown %'})
                    fig_dd.update_layout(height=300, template="plotly_white", margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig_dd, use_container_width=True)

        # Export Options
        st.markdown('<div class="section-header"><h4 style="margin: 0;">📥 Export Data & Summaries</h4></div>', unsafe_allow_html=True)
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button("📄 Download CSV Data", data.to_csv().encode('utf-8'), f"{stock_symbol}_data.csv", "text/csv", use_container_width=True)
        with exp_col2:
            summary_dict = {"symbol": stock_symbol, "price": current_price, "rsi": rsi_value, "date": str(datetime.now().date())}
            st.download_button("📊 Download Summary JSON", json.dumps(summary_dict, indent=2).encode('utf-8'), f"{stock_symbol}_summary.json", "application/json", use_container_width=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
        <h3>🚀 Ready to Analyze</h3>
        <p style="color: #6b7280; max-width: 600px; margin: 0 auto;">
            Choose a stock from the dropdown or type a stock symbol in the sidebar, configure your settings, and click <b>'Run Analysis'</b> to generate indicators, risk metrics, and market sentiment.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">
<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <p style="margin: 0.5rem 0;">Professional stock analysis platform • Not financial advice</p>
    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">© 2026 Advanced Stock Analyzer Pro</p>
</div>
""", unsafe_allow_html=True)