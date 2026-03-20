import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')

# Streamlit App Configuration
st.set_page_config(page_title="📈 Advanced Stock Analyzer", layout="wide")

# 🔥 API Keys (Loaded from secrets.toml)
NEWSAPI_KEY = st.secrets.get("newsapi", {}).get("api_key")
FINNHUB_KEY = st.secrets.get("finnhub", {}).get("api_key")
ALPHA_VANTAGE_KEY = st.secrets.get("alphavantage", {}).get("api_key")

# Professional CSS
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
    .trend-up { color: #10b981; font-weight: bold; }
    .trend-down { color: #ef4444; font-weight: bold; }
    .trend-neutral { color: #6b7280; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
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

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 📊 Stock Selection")
    stock_symbol = st.text_input("Stock Symbol:", "AAPL").upper()
    
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
    
    # API Status
    st.markdown("#### 🔌 API Status")
    api_status = {}
    if NEWSAPI_KEY:
        api_status["News"] = "✅ Connected"
    else:
        api_status["News"] = "⚠️ No Key"
    
    if FINNHUB_KEY:
        api_status["Finnhub"] = "✅ Connected"
    else:
        api_status["Finnhub"] = "⚠️ No Key"
    
    for api, status in api_status.items():
        st.caption(f"{api}: {status}")
    
    st.markdown("---")
    
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
        st.session_state.selected_stock = stock_symbol

# Initialize session state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Fetch and process stock data
# Replace the fetch_stock_data function (around line 87) with this corrected version:
# Replace the ENTIRE fetch_stock_data function with this corrected version:

def fetch_stock_data(symbol, start, end):
    """Fetch stock data with error handling"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get historical data
        data = yf.download(symbol, start=start, end=end, progress=False)
        
        if data.empty:
            return None, None
        
        # yfinance returns MultiIndex columns - flatten them
        if isinstance(data.columns, pd.MultiIndex):
            # Fix: Flatten the MultiIndex columns
            data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in data.columns]
        
        # Check if we have the required columns (some might have different names)
        column_mapping = {}
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            # Check for exact match first
            if col in data.columns:
                column_mapping[col] = col
            else:
                # Check for variations (case-insensitive)
                matches = [c for c in data.columns if col.lower() in c.lower()]
                if matches:
                    column_mapping[col] = matches[0]
                else:
                    # If column not found, create it with NaN
                    data[col] = np.nan
                    column_mapping[col] = col
        
        # Rename columns to standard names for easier processing
        rename_dict = {v: k for k, v in column_mapping.items() if k != v}
        if rename_dict:
            data = data.rename(columns=rename_dict)
        
        # Convert numeric columns - handle both Series and single values
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                # Handle if it's already numeric
                if not pd.api.types.is_numeric_dtype(data[col]):
                    # Convert to numeric, coerce errors
                    data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by date
        data = data.sort_index()
        
        # Clean data - forward fill then drop remaining NaN
        data = data.ffill().bfill()
        
        # Calculate technical indicators
        # Moving Averages
        if 'Close' in data.columns:
            close_series = data['Close']
            data['SMA_50'] = close_series.rolling(window=50, min_periods=1).mean()
            data['SMA_200'] = close_series.rolling(window=200, min_periods=1).mean()
            data['EMA_20'] = close_series.ewm(span=20, adjust=False).mean()
            
            # RSI Calculation
            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            data['RSI'] = calculate_rsi(close_series)
            
            # MACD Calculation
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
            data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
            
            # Volatility
            data['daily_returns'] = close_series.pct_change()
            data['volatility'] = data['daily_returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
        
        # Volume indicators
        if 'Volume' in data.columns and 'Close' in data.columns:
            try:
                # VWAP calculation
                typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                data['VWAP'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            except:
                data['VWAP'] = np.nan
        
        return data, info
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None


# News Sentiment Analysis
def get_news_sentiment(symbol):
    """Fetch and analyze news sentiment for the stock"""
    news_data = []
    
    # Try NewsAPI first
    if NEWSAPI_KEY:
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWSAPI_KEY}&language=en&sortBy=publishedAt&pageSize=10"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                for article in articles:
                    news_data.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', '')
                    })
        except:
            pass
    
    # Fallback to Finnhub
    if not news_data and FINNHUB_KEY:
        try:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={(date.today() - timedelta(days=30)).strftime('%Y-%m-%d')}&to={date.today().strftime('%Y-%m-%d')}&token={FINNHUB_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json()[:10]  # Limit to 10 articles
                for article in articles:
                    news_data.append({
                        'title': article.get('headline', ''),
                        'description': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'published': datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d'),
                        'source': article.get('source', '')
                    })
        except:
            pass
    
    # Simple sentiment analysis based on keywords
    if news_data:
        positive_keywords = ['up', 'bullish', 'growth', 'profit', 'gain', 'positive', 'buy', 'strong']
        negative_keywords = ['down', 'bearish', 'loss', 'decline', 'negative', 'sell', 'weak', 'risk']
        
        sentiments = []
        for news in news_data:
            title_desc = f"{news['title']} {news['description']}".lower()
            pos_count = sum(1 for word in positive_keywords if word in title_desc)
            neg_count = sum(1 for word in negative_keywords if word in title_desc)
            
            if pos_count > neg_count:
                sentiment = 1
            elif neg_count > pos_count:
                sentiment = -1
            else:
                sentiment = 0
            
            news['sentiment'] = sentiment
            sentiments.append(sentiment)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return news_data, avg_sentiment
    
    return [], 0

# Technical Pattern Recognition
# In the detect_technical_patterns function (around line 159), update these lines:
def detect_technical_patterns(data):
    """Identify technical chart patterns"""
    patterns = []
    
    if len(data) < 50:
        return patterns
    
    close_prices = data['Close'].values
    high_prices = data['High'].values
    low_prices = data['Low'].values
    
    # Double Top/Bottom detection
    for i in range(50, len(close_prices) - 20):
        # Look for resistance/support levels
        recent_highs = high_prices[i-20:i]
        recent_lows = low_prices[i-20:i]
        
        # Double Top pattern (resistance)
        if len(recent_highs) >= 10:
            resistance_level = np.max(recent_highs[-10:])
            if high_prices[i] >= resistance_level * 0.98:
                patterns.append({
                    'pattern': 'Double Top',
                    'date': data.index[i],
                    'price': high_prices[i],
                    'confidence': 'Medium'
                })
        
        # Double Bottom pattern (support)
        if len(recent_lows) >= 10:
            support_level = np.min(recent_lows[-10:])
            if low_prices[i] <= support_level * 1.02:
                patterns.append({
                    'pattern': 'Double Bottom',
                    'date': data.index[i],
                    'price': low_prices[i],
                    'confidence': 'Medium'
                })
    
    # Trend detection
    sma_50 = data['SMA_50'].values
    sma_200 = data['SMA_200'].values
    
    # Golden Cross
    if len(sma_50) > 20 and len(sma_200) > 20:
        if sma_50[-1] > sma_200[-1] and sma_50[-5] <= sma_200[-5]:
            patterns.append({
                'pattern': 'Golden Cross',
                'date': data.index[-1],
                'price': close_prices[-1],
                'confidence': 'High'
            })
        
        # Death Cross
        if sma_50[-1] < sma_200[-1] and sma_50[-5] >= sma_200[-5]:
            patterns.append({
                'pattern': 'Death Cross',
                'date': data.index[-1],
                'price': close_prices[-1],
                'confidence': 'High'
            })
    
    return patterns[-5:]  # Return last 5 patterns

# Risk Assessment Metrics
def calculate_risk_metrics(data):
    """Calculate various risk metrics"""
    if len(data) < 30:
        return {}
    
    returns = data['daily_returns'].dropna()
    
    metrics = {
        'annual_volatility': returns.std() * np.sqrt(252) * 100,
        'max_drawdown': (data['Close'] / data['Close'].cummax() - 1).min() * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
        'var_95': np.percentile(returns, 5) * 100,  # 5% VaR
        'beta': 1.0,  # Simplified beta (would need market data for real calculation)
        'sortino_ratio': (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() > 0 else 0
    }
    
    return metrics

# Volatility Forecasting
def forecast_volatility(data):
    """Simple GARCH-like volatility forecasting"""
    if len(data) < 60:
        return None, None
    
    returns = data['daily_returns'].dropna().values[-60:]
    
    # Simple moving average of squared returns (simplified GARCH)
    forecast_window = 5
    squared_returns = returns ** 2
    forecast_vol = np.mean(squared_returns[-20:]) * np.sqrt(252) * 100
    
    # Trend in volatility
    recent_vol = data['volatility'].dropna().values[-20:]
    vol_trend = 'increasing' if recent_vol[-1] > recent_vol[0] else 'decreasing'
    
    return forecast_vol, vol_trend

# Portfolio Optimization Suggestions
def get_portfolio_suggestions(data, symbol):
    """Generate portfolio optimization suggestions"""
    suggestions = []
    
    if len(data) < 100:
        return suggestions
    
    current_price = data['Close'].iloc[-1]
    sma_50 = data['SMA_50'].iloc[-1]
    sma_200 = data['SMA_200'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    
    # Buy/Sell/Hold recommendations
    if current_price > sma_50 > sma_200 and rsi < 70:
        suggestions.append({
            'action': 'BUY',
            'reason': 'Strong uptrend with reasonable RSI',
            'confidence': 'High',
            'suggested_allocation': '3-5% of portfolio'
        })
    elif current_price < sma_200 and rsi < 30:
        suggestions.append({
            'action': 'BUY',
            'reason': 'Oversold with long-term support',
            'confidence': 'Medium',
            'suggested_allocation': '2-3% of portfolio'
        })
    elif current_price > sma_50 and rsi > 70:
        suggestions.append({
            'action': 'SELL',
            'reason': 'Overbought condition',
            'confidence': 'Medium',
            'suggested_allocation': 'Reduce position by 50%'
        })
    elif current_price < sma_50 < sma_200:
        suggestions.append({
            'action': 'SELL',
            'reason': 'Strong downtrend',
            'confidence': 'High',
            'suggested_allocation': 'Exit position'
        })
    else:
        suggestions.append({
            'action': 'HOLD',
            'reason': 'Mixed signals, wait for confirmation',
            'confidence': 'Low',
            'suggested_allocation': 'Maintain current position'
        })
    
    # Risk-based suggestions
    risk_metrics = calculate_risk_metrics(data)
    volatility = risk_metrics.get('annual_volatility', 0)
    
    if volatility > 40:
        suggestions.append({
            'action': 'REDUCE',
            'reason': 'High volatility stock',
            'confidence': 'High',
            'suggested_allocation': 'Limit to 1-2% of portfolio'
        })
    elif volatility < 20:
        suggestions.append({
            'action': 'INCREASE',
            'reason': 'Low volatility, stable returns',
            'confidence': 'Medium',
            'suggested_allocation': 'Can allocate 5-7% of portfolio'
        })
    
    return suggestions

# Main Analysis Execution
if st.session_state.run_analysis:
    if start_date >= end_date:
        st.error("❌ Invalid date range. The start date must be before the end date.")
    else:
        with st.spinner(f"🔍 Analyzing {stock_symbol}..."):
            
            # Fetch data
            data, info = fetch_stock_data(stock_symbol, start_date, end_date)
            
            if data is None or data.empty:
                st.error("⚠️ No data found. Please check the stock symbol or date range.")
            else:
                # Stock Overview Section
                st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Stock Overview</h3></div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    current_price = data['Close'].iloc[-1]
                    price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                    st.metric(
                        "Current Price", 
                        f"${current_price:.2f}",
                        f"{price_change:+.2f}%"
                    )
                
                with col2:
                    if info and 'marketCap' in info:
                        market_cap = info['marketCap'] / 1e9
                        st.metric("Market Cap", f"${market_cap:.1f}B")
                    else:
                        st.metric("Market Cap", "N/A")
                
                with col3:
                    volume = data['Volume'].iloc[-1]
                    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                    st.metric(
                        "Volume", 
                        f"{volume:,.0f}",
                        f"{(volume/avg_volume - 1)*100:+.1f}% vs avg"
                    )
                
                with col4:
                    rsi_value = data['RSI'].iloc[-1]
                    rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                    st.metric("RSI", f"{rsi_value:.1f}", rsi_status)
                
                # Main Tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Price Analysis", 
                    "🤖 AI Insights", 
                    "📰 News & Sentiment",
                    "⚖️ Risk & Portfolio"
                ])
                
                with tab1:
                    # Price and Volume Chart
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Candlestick Chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="Price",
                            increasing_line_color='#10b981',
                            decreasing_line_color='#ef4444'
                        ))
                        
                        # Moving averages
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data['SMA_50'],
                            mode="lines", name="50-Day MA",
                            line=dict(color='#f59e0b', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data['SMA_200'],
                            mode="lines", name="200-Day MA",
                            line=dict(color='#8b5cf6', width=2)
                        ))
                        
                        # Bollinger Bands
                        # In the main chart section (around line 283), update the Bollinger Bands section:

                        # Bollinger Bands
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data['BB_upper'],
                            mode="lines", name="BB Upper",
                            line=dict(color='rgba(59, 130, 246, 0.3)', width=1),
                            showlegend=False
                        ))

                        fig.add_trace(go.Scatter(
                            x=data.index, y=data['BB_lower'],
                            mode="lines", name="BB Lower",
                            line=dict(color='rgba(59, 130, 246, 0.3)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(59, 130, 246, 0.1)',
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            title=f"{stock_symbol} Price Chart with Indicators",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            xaxis_rangeslider_visible=False,
                            height=500,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Technical Indicators")
                        
                        # RSI Gauge
                        rsi_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=rsi_value,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "RSI"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': rsi_value
                                }
                            }
                        ))
                        rsi_gauge.update_layout(height=250)
                        st.plotly_chart(rsi_gauge, use_container_width=True)
                        
                        # MACD
                        # Around line 321, update the MACD calculation:
                        macd_value = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
                        st.metric("MACD", f"{macd_value:.4f}", 
                                "Bullish" if macd_value > 0 else "Bearish")
                        
                        # Volume
                        volume_ratio = volume / avg_volume
                        st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
                        
                        # Volatility
                        volatility = data['volatility'].iloc[-1] * 100
                        st.metric("20-Day Volatility", f"{volatility:.1f}%")
                
                with tab2:
                    st.markdown('<div class="section-header"><h4 style="margin: 0;">🤖 AI-Powered Analysis</h4></div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pattern Recognition
                        if show_patterns:
                            st.markdown("#### 🔍 Technical Patterns")
                            patterns = detect_technical_patterns(data)
                            
                            if patterns:
                                for pattern in patterns:
                                    with st.container():
                                        col_a, col_b = st.columns([3, 1])
                                        with col_a:
                                            st.markdown(f"**{pattern['pattern']}**")
                                            st.caption(f"Date: {pattern['date'].strftime('%Y-%m-%d')}")
                                        with col_b:
                                            confidence_color = {
                                                'High': '🟢',
                                                'Medium': '🟡',
                                                'Low': '🔴'
                                            }.get(pattern['confidence'], '⚪')
                                            st.markdown(f"{confidence_color} {pattern['confidence']}")
                            else:
                                st.info("No strong technical patterns detected.")
                        
                        # Volatility Forecasting
                        st.markdown("#### 📊 Volatility Forecast")
                        forecast_vol, vol_trend = forecast_volatility(data)
                        
                        if forecast_vol:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("5-Day Vol Forecast", f"{forecast_vol:.1f}%")
                            with col_b:
                                trend_icon = "📈" if vol_trend == 'increasing' else "📉"
                                st.metric("Trend", f"{trend_icon} {vol_trend}")
                    
                    with col2:
                        # Portfolio Suggestions
                        st.markdown("#### 🎯 Portfolio Optimization")
                        suggestions = get_portfolio_suggestions(data, stock_symbol)
                        
                        for suggestion in suggestions:
                            with st.container():
                                st.markdown(f"##### {suggestion['action']}")
                                st.caption(suggestion['reason'])
                                st.progress({
                                    'High': 0.9,
                                    'Medium': 0.6,
                                    'Low': 0.3
                                }.get(suggestion['confidence'], 0.5))
                                st.caption(f"Suggested: {suggestion['suggested_allocation']}")
                        
                        # Trend Analysis
                        st.markdown("#### 📈 Trend Analysis")
                        current_price = data['Close'].iloc[-1]
                        sma_50 = data['SMA_50'].iloc[-1]
                        sma_200 = data['SMA_200'].iloc[-1]
                        
                        if current_price > sma_50 > sma_200:
                            st.success("**Strong Uptrend** 📈")
                            st.caption("Price above both moving averages")
                        elif current_price < sma_50 < sma_200:
                            st.error("**Strong Downtrend** 📉")
                            st.caption("Price below both moving averages")
                        else:
                            st.warning("**Consolidation/Ranging** ↔️")
                            st.caption("Mixed signals, watch for breakout")
                
                with tab3:
                    if include_news:
                        st.markdown('<div class="section-header"><h4 style="margin: 0;">📰 News & Market Sentiment</h4></div>', unsafe_allow_html=True)
                        
                        # Fetch news
                        news_articles, avg_sentiment = get_news_sentiment(stock_symbol)
                        
                        if news_articles:
                            # Sentiment Summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                sentiment_score = avg_sentiment * 100
                                st.metric("Overall Sentiment", 
                                         f"{sentiment_score:.0f}",
                                         "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral")
                            
                            with col2:
                                positive_articles = sum(1 for n in news_articles if n['sentiment'] > 0)
                                st.metric("Positive Articles", positive_articles)
                            
                            with col3:
                                negative_articles = sum(1 for n in news_articles if n['sentiment'] < 0)
                                st.metric("Negative Articles", negative_articles)
                            
                            # News Articles
                            st.markdown("#### Recent News")
                            for article in news_articles[:5]:  # Show top 5
                                with st.container():
                                    sentiment_icon = "🟢" if article['sentiment'] > 0 else "🔴" if article['sentiment'] < 0 else "⚪"
                                    st.markdown(f"##### {sentiment_icon} {article['title']}")
                                    if article.get('description'):
                                        st.caption(article['description'])
                                    
                                    col_a, col_b, col_c = st.columns([2, 1, 1])
                                    with col_a:
                                        st.caption(f"Source: {article['source']}")
                                    with col_b:
                                        st.caption(f"Date: {article['published'][:10] if len(article['published']) > 10 else article['published']}")
                                    with col_c:
                                        if article['url'] != '#':
                                            st.link_button("Read More", article['url'])
                                    
                                    st.divider()
                        else:
                            st.info("No recent news articles found. Check your API keys or try another stock.")
                    else:
                        st.info("News sentiment analysis is disabled. Enable it in the sidebar settings.")
                
                with tab4:
                    st.markdown('<div class="section-header"><h4 style="margin: 0;">⚖️ Risk Assessment & Portfolio Metrics</h4></div>', unsafe_allow_html=True)
                    
                    if risk_analysis:
                        risk_metrics = calculate_risk_metrics(data)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Risk Metrics")
                            
                            # Volatility
                            volatility = risk_metrics.get('annual_volatility', 0)
                            vol_level = "High" if volatility > 30 else "Medium" if volatility > 15 else "Low"
                            st.metric("Annual Volatility", f"{volatility:.1f}%", vol_level)
                            
                            # Max Drawdown
                            max_dd = risk_metrics.get('max_drawdown', 0)
                            st.metric("Max Drawdown", f"{max_dd:.1f}%")
                            
                            # Sharpe Ratio
                            sharpe = risk_metrics.get('sharpe_ratio', 0)
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                            
                            # Value at Risk
                            var_95 = risk_metrics.get('var_95', 0)
                            st.metric("5% Daily VaR", f"{var_95:.1f}%")
                        
                        with col2:
                            st.markdown("#### 📈 Risk Visualization")
                            
                            # Drawdown chart
                            drawdown = (data['Close'] / data['Close'].cummax() - 1) * 100
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=data.index,
                                y=drawdown,
                                mode='lines',
                                name='Drawdown',
                                fill='tozeroy',
                                fillcolor='rgba(239, 68, 68, 0.3)',
                                line=dict(color='#ef4444')
                            ))
                            fig_dd.update_layout(
                                title="Drawdown Analysis",
                                xaxis_title="Date",
                                yaxis_title="Drawdown (%)",
                                height=300,
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_dd, use_container_width=True)
                            
                            # Risk Rating
                            risk_score = (volatility / 50 * 0.4 + abs(max_dd) / 50 * 0.3 + 
                                        (1 - min(sharpe, 2)/2) * 0.3) * 100
                            risk_rating = "High" if risk_score > 70 else "Medium" if risk_score > 40 else "Low"
                            
                            st.metric("Overall Risk Rating", risk_rating, f"Score: {risk_score:.0f}/100")
                    
                    # Portfolio Allocation Advice
                    st.markdown("#### 🎯 Allocation Strategy")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("##### Conservative")
                        st.caption("Low risk tolerance")
                        st.metric("Allocation", "1-2%")
                        st.caption("Suitable for: Retirees, risk-averse")
                    
                    with col2:
                        st.markdown("##### Balanced")
                        st.caption("Moderate risk tolerance")
                        st.metric("Allocation", "3-5%")
                        st.caption("Suitable for: Most investors")
                    
                    with col3:
                        st.markdown("##### Aggressive")
                        st.caption("High risk tolerance")
                        st.metric("Allocation", "5-7%")
                        st.caption("Suitable for: Growth investors")
                
                # Data Download Section
                st.markdown('<div class="section-header"><h4 style="margin: 0;">📥 Export Data</h4></div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = data.to_csv().encode("utf-8")
                    st.download_button(
                        label="📄 Download Stock Data (CSV)",
                        data=csv_data,
                        file_name=f"{stock_symbol}_stock_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Export analysis summary
                    summary = {
                        "stock": stock_symbol,
                        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                        "current_price": float(data['Close'].iloc[-1]),
                        "recommendation": suggestions[0]['action'] if suggestions else "HOLD",
                        "risk_score": risk_score if 'risk_score' in locals() else 0,
                        "volatility": float(volatility) if 'volatility' in locals() else 0
                    }
                    json_data = json.dumps(summary, indent=2)
                    st.download_button(
                        label="📊 Download Analysis Summary (JSON)",
                        data=json_data,
                        file_name=f"{stock_symbol}_analysis_summary.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    if st.button("🔄 New Analysis", use_container_width=True):
                        st.session_state.run_analysis = False
                        st.rerun()

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">

<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
        <span>📈 <strong>Technical Analysis</strong></span>
        <span>🤖 <strong>AI Insights</strong></span>
        <span>📰 <strong>News Sentiment</strong></span>
        <span>⚖️ <strong>Risk Assessment</strong></span>
        <span>🎯 <strong>Portfolio Optimization</strong></span>
    </div>
    <p style="margin: 0.5rem 0; color: #4b5563;">
        Professional stock analysis tool for informed investment decisions • Not financial advice
    </p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">
        © 2024 Advanced Stock Analyzer Pro | Data provided by Yahoo Finance
    </p>
</div>
""", unsafe_allow_html=True)

# Initial state message
if not st.session_state.run_analysis:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
        <h3 style="color: #4b5563; margin-bottom: 1rem;">🚀 Ready to Analyze</h3>
        <p style="color: #6b7280; max-width: 600px; margin: 0 auto;">
            Enter a stock symbol in the sidebar and configure your analysis settings to begin.
            Get comprehensive insights including technical analysis, news sentiment, and portfolio recommendations.
        </p>
        <div style="margin-top: 2rem; color: #9ca3af; font-size: 0.9rem;">
            <p>📋 Supported Features:</p>
            <div style="display: inline-flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin-top: 1rem;">
                <span>• Real-time Technical Indicators</span>
                <span>• News Sentiment Analysis</span>
                <span>• Pattern Recognition</span>
                <span>• Volatility Forecasting</span>
                <span>• Risk Assessment</span>
                <span>• Portfolio Optimization</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)