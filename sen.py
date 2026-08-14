from collections import Counter
from datetime import datetime
import json
import re
import warnings
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# Set up Streamlit App configuration
st.set_page_config(page_title="🎭 Advanced Sentiment Analyzer Pro", layout="wide")

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
        border-left: 4px solid #3b82f6;
    }
    .section-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
    }
    .sentiment-positive { color: #10b981; font-weight: bold; }
    .sentiment-negative { color: #ef4444; font-weight: bold; }
    .sentiment-neutral { color: #6b7280; font-weight: bold; }
    .emotion-card {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎭 Advanced Sentiment Analyzer Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Multi-dimensional Text Analysis with Aspect Extraction, Emotion Detection & AI Insights
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 📊 Data Source")
    source_type = st.radio("Choose source:", ["📁 Upload CSV", "📋 Sample Data"], index=1)
    
    st.markdown("---")
    st.markdown("#### 🔧 Analysis Settings")
    include_aspects = st.checkbox("🔍 Aspect-based Analysis", value=True)
    include_emotions = st.checkbox("😊 Emotion Detection", value=True)
    include_topics = st.checkbox("📊 Topic Modeling", value=True)
    include_trends = st.checkbox("📈 Trend Forecasting", value=True)
    use_llm_ensemble = st.checkbox("🤖 AI Ensemble", value=False)
    
    st.markdown("---")
    st.markdown("#### 🎯 Advanced Features")
    analysis_depth = st.select_slider(
        "Analysis Depth:",
        options=["Basic", "Standard", "Advanced", "Expert"],
        value="Advanced"
    )
    min_confidence = st.slider("Minimum Confidence", 0.5, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
        st.session_state.analysis_completed = False  # Trigger re-analysis

# Initialize session state flags
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False

@st.cache_data
def load_sample_data() -> dict:
    """Load diverse sample datasets with timestamps and categories."""
    return {
        "📱 Product Reviews": pd.DataFrame({
            "text": [
                "The camera quality is excellent but battery life needs improvement.",
                "Great display but the software is buggy and crashes frequently.",
                "Love the design and performance, worth every penny!",
                "Customer service was terrible, took weeks to resolve my issue.",
                "Fast shipping and perfect packaging, very satisfied!",
                "The phone overheats during gaming sessions.",
                "Amazing sound quality from the speakers, very impressed.",
                "The screen cracked too easily, not durable at all.",
                "Best purchase I've made this year, highly recommended!",
                "Charging port stopped working after 2 months."
            ],
            "date": pd.date_range('2026-01-01', periods=10, freq='D'),
            "category": ["Electronics", "Electronics", "Electronics", "Service", "Service", 
                        "Electronics", "Electronics", "Electronics", "Electronics", "Electronics"]
        }),
        "🏨 Hotel Reviews": pd.DataFrame({
            "text": [
                "Beautiful location but rooms were not clean.",
                "Excellent breakfast and friendly staff made our stay wonderful.",
                "Room service was slow and food was cold when it arrived.",
                "The pool area was amazing, kids loved it!",
                "Noise from construction nearby ruined our sleep.",
                "Spacious rooms with modern amenities, will come back!",
                "Reception was rude and unhelpful.",
                "Great value for money, clean and comfortable.",
                "WiFi was terrible throughout the hotel.",
                "Perfect romantic getaway, beautiful sunset views."
            ],
            "date": pd.date_range('2026-01-01', periods=10, freq='D'),
            "category": ["Accommodation", "Food", "Service", "Amenities", "Noise",
                        "Rooms", "Service", "Value", "Internet", "Location"]
        }),
        "🎬 Movie Reviews": pd.DataFrame({
            "text": [
                "The plot was predictable but acting was superb.",
                "Amazing cinematography and soundtrack, a visual masterpiece.",
                "Dialogue felt forced and characters were not believable.",
                "Best movie of the year, edge of the seat thriller!",
                "Waste of time, terrible direction and poor story.",
                "Emotional and heartwarming, brought tears to my eyes.",
                "Special effects were overused, distracted from the story.",
                "Perfect balance of comedy and drama, loved every minute.",
                "Plot holes ruined an otherwise good movie.",
                "Masterful storytelling with brilliant performances."
            ],
            "date": pd.date_range('2026-01-01', periods=10, freq='D'),
            "category": ["Plot", "Cinematography", "Acting", "Overall", "Direction",
                        "Emotion", "Effects", "Genre", "Plot", "Storytelling"]
        }),
        "🛒 E-commerce Feedback": pd.DataFrame({
            "text": [
                "Product arrived damaged but customer support was helpful.",
                "Excellent quality, exactly as described in the pictures.",
                "Shipping took too long, package was delayed by a week.",
                "Love the product, fits perfectly and looks great!",
                "Material feels cheap, not worth the price.",
                "Easy checkout process and fast delivery.",
                "Wrong color was sent, had to return and wait again.",
                "Great value, better than expected!",
                "Return policy is confusing and customer service was unresponsive.",
                "Perfect gift, recipient was very happy with it."
            ],
            "date": pd.date_range('2026-01-01', periods=10, freq='D'),
            "category": ["Delivery", "Quality", "Shipping", "Satisfaction", "Value",
                        "Experience", "Mistakes", "Value", "Policy", "Gifting"]
        })
    }

# Load data based on user selection with validation
data = None
text_column = "text"
if source_type == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            temp_df = pd.read_csv(uploaded_file)
            st.markdown("### ⚙️ Map Columns")
            cols = list(temp_df.columns)
            detected_text_col = next((c for c in cols if 'text' in c.lower() or 'review' in c.lower() or 'comment' in c.lower()), cols[0])
            text_column = st.selectbox("Select Text Column:", cols, index=cols.index(detected_text_col))
            
            date_col = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower()), None)
            selected_date_col = st.selectbox("Select Date Column (Optional):", ["None"] + cols, index=0 if not date_col else cols.index(date_col) + 1)
            
            category_col = next((c for c in cols if 'category' in c.lower() or 'type' in c.lower() or 'sentiment' in c.lower()), None)
            selected_cat_col = st.selectbox("Select Category Column (Optional):", ["None"] + cols, index=0 if not category_col else cols.index(category_col) + 1)
            
            data = pd.DataFrame()
            data['text'] = temp_df[text_column].astype(str)
            if selected_date_col != "None":
                data['date'] = pd.to_datetime(temp_df[selected_date_col], errors='coerce')
            if selected_cat_col != "None":
                data['category'] = temp_df[selected_cat_col]
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
else:
    sample_datasets = load_sample_data()
    dataset_choice = st.selectbox("Choose a sample dataset:", list(sample_datasets.keys()))
    data = sample_datasets[dataset_choice]

@st.cache_data
def preprocess_text(text: str) -> str:
    """Clean and preprocess raw input text strings."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

def extract_aspects_and_sentiment(text: str) -> list:
    """Extract aspects and compute associated sentiments."""
    aspects = []
    aspect_keywords = {
        'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money'],
        'quality': ['quality', 'durable', 'strong', 'weak', 'break', 'last', 'material'],
        'service': ['service', 'support', 'help', 'response', 'staff', 'representative'],
        'delivery': ['delivery', 'shipping', 'arrive', 'package', 'fast', 'slow', 'delay'],
        'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'work', 'function'],
        'design': ['design', 'look', 'appearance', 'style', 'color', 'size', 'shape'],
        'features': ['feature', 'function', 'option', 'capability', 'ability', 'tool'],
        'usability': ['easy', 'difficult', 'simple', 'complex', 'user-friendly', 'interface']
    }
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in text.lower():
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                sentiment = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
                aspects.append({
                    'aspect': aspect,
                    'keyword': keyword,
                    'sentiment': sentiment,
                    'polarity': polarity,
                    'confidence': abs(polarity)
                })
                break
    return aspects

def detect_emotion(text: str) -> tuple:
    """Detect dominant emotions using a lexicon-based approach."""
    emotion_lexicon = {
        'joy': ['love', 'happy', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 
                'perfect', 'best', 'awesome', 'enjoy', 'delight', 'pleasure'],
        'anger': ['hate', 'angry', 'terrible', 'awful', 'horrible', 'worst', 'frustrated',
                 'annoyed', 'mad', 'rage', 'irritated'],
        'sadness': ['sad', 'disappoint', 'unhappy', 'regret', 'sorry', 'cry', 'tear',
                   'depressed', 'miserable', 'heartbreak'],
        'surprise': ['surprise', 'shock', 'wow', 'unexpected', 'astonish', 'amaze'],
        'fear': ['fear', 'scared', 'afraid', 'worry', 'anxious', 'nervous'],
        'trust': ['trust', 'reliable', 'dependable', 'honest', 'sincere', 'faith'],
        'disgust': ['disgust', 'gross', 'nasty', 'revolting', 'sickening', 'vile']
    }
    emotion_scores = {emotion: 0 for emotion in emotion_lexicon.keys()}
    words = text.lower().split()
    for word in words:
        for emotion, keywords in emotion_lexicon.items():
            if word in keywords:
                emotion_scores[emotion] += 1
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        if dominant_emotion[1] > 0:
            return dominant_emotion[0], emotion_scores
    return "neutral", emotion_scores

def extract_topics(texts: list, n_topics: int = 3) -> list:
    """Extract frequent terms to approximate topics."""
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(20)
    topics = []
    if len(common_words) >= n_topics * 3:
        for i in range(n_topics):
            start_idx = i * 3
            end_idx = start_idx + 3
            topic_words = [word for word, _ in common_words[start_idx:end_idx]]
            topics.append({
                'topic_id': i+1,
                'keywords': topic_words,
                'frequency': sum(freq for _, freq in common_words[start_idx:end_idx])
            })
    return topics

def ensemble_sentiment_analysis(text: str) -> dict:
    """Ensemble sentiment scoring across multiple approaches."""
    blob = TextBlob(text)
    textblob_score = blob.sentiment.polarity
    textblob_sentiment = "positive" if textblob_score > 0.1 else "negative" if textblob_score < -0.1 else "neutral"
    
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'happy']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappoint']
    rule_score = 0
    words = text.lower().split()
    for word in words:
        if word in positive_words:
            rule_score += 1
        elif word in negative_words:
            rule_score -= 1
    rule_sentiment = "positive" if rule_score > 0 else "negative" if rule_score < 0 else "neutral"
    
    length_score = min(len(text.split()) / 50, 1.0)
    length_sentiment = "positive" if length_score > 0.7 else "neutral"
    
    votes = {'textblob': textblob_sentiment, 'rule': rule_sentiment, 'length': length_sentiment}
    vote_counts = Counter(votes.values())
    final_sentiment = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[final_sentiment] / len(votes)
    
    return {
        'sentiment': final_sentiment,
        'confidence': confidence,
        'scores': {'textblob': textblob_score, 'rule': rule_score, 'length': length_score},
        'votes': votes
    }

def forecast_sentiment_trend(sentiment_data: list, dates: list) -> tuple:
    """Forecast short-term sentiment trends."""
    if len(sentiment_data) < 5:
        return None, None
    sentiment_numeric = [1 if s == 'positive' else (-1 if s == 'negative' else 0) for s in sentiment_data]
    window = min(3, len(sentiment_numeric) // 2)
    if window > 1:
        trend = np.convolve(sentiment_numeric, np.ones(window)/window, mode='valid')
        if len(trend) > 1:
            current_trend = "improving" if trend[-1] > trend[-2] else "declining" if trend[-1] < trend[-2] else "stable"
            forecast = "positive" if trend[-1] > 0.3 else "negative" if trend[-1] < -0.3 else "neutral"
            return current_trend, forecast
    return "stable", "neutral"

if data is not None and 'text' in data.columns:
    if st.session_state.run_analysis:
        # Check session state persistence to avoid redundant re-computations when switching tabs
        if not st.session_state.get('analysis_completed', False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preprocessing text inputs...")
            progress_bar.progress(30)
            data['cleaned_text'] = data['text'].apply(preprocess_text)
            
            status_text.text("Running ensemble sentiment analysis...")
            progress_bar.progress(60)
            
            analysis_results = []
            for idx, row in data.iterrows():
                text = row['cleaned_text']
                basic_sentiment = ensemble_sentiment_analysis(text)
                result = {
                    'original_text': row['text'],
                    'cleaned_text': text,
                    'sentiment': basic_sentiment['sentiment'],
                    'confidence': basic_sentiment['confidence'],
                    'polarity': basic_sentiment['scores']['textblob']
                }
                if include_aspects:
                    aspects = extract_aspects_and_sentiment(text)
                    result['aspects'] = aspects
                    result['aspect_count'] = len(aspects)
                if include_emotions:
                    emotion, emotion_scores = detect_emotion(text)
                    result['dominant_emotion'] = emotion
                    result['emotion_scores'] = emotion_scores
                if 'date' in data.columns:
                    result['date'] = row['date']
                if 'category' in data.columns:
                    result['category'] = row['category']
                analysis_results.append(result)
            
            st.session_state.results_df = pd.DataFrame(analysis_results)
            st.session_state.analysis_completed = True
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

        results_df = st.session_state.results_df

        # Filter by confidence threshold
        results_df = results_df[results_df['confidence'] >= min_confidence]

        st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Analysis Overview</h3></div>', unsafe_allow_html=True)
        
        positive_count = len(results_df[results_df['sentiment'] == 'positive'])
        negative_count = len(results_df[results_df['sentiment'] == 'negative'])
        neutral_count = len(results_df[results_df['sentiment'] == 'neutral'])
        total_analyzed = len(results_df)
        avg_confidence = results_df['confidence'].mean() if total_analyzed > 0 else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positive", positive_count)
        col2.metric("Negative", negative_count)
        col3.metric("Neutral", neutral_count)
        col4.metric("Avg Confidence", f"{avg_confidence:.1%}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Detailed Results", 
            "📈 Visual Analytics", 
            "🎯 Aspect Analysis",
            "😊 Emotion Insights",
            "📊 Advanced Analytics"
        ])
        
        with tab1:
            display_cols = ['original_text', 'sentiment', 'confidence', 'polarity']
            if 'category' in results_df.columns:
                display_cols.insert(1, 'category')
            
            st.dataframe(
                results_df[display_cols],
                column_config={
                    "original_text": "Original Text",
                    "sentiment": st.column_config.SelectboxColumn("Sentiment", options=["positive", "negative", "neutral"]),
                    "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=1),
                    "polarity": st.column_config.NumberColumn("Polarity", format="%.2f")
                },
                use_container_width=True,
                height=400
            )
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                sentiment_counts = results_df['sentiment'].value_counts()
                fig1 = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#6b7280'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.histogram(
                    results_df,
                    x='confidence',
                    nbins=20,
                    title="Confidence Score Distribution",
                    color='sentiment',
                    color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#6b7280'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            if 'date' in results_df.columns:
                st.markdown("#### 📈 Sentiment Trends Over Time")
                results_df['date'] = pd.to_datetime(results_df['date'])
                daily_sentiment = results_df.groupby(results_df['date'].dt.date)['sentiment'].apply(
                    lambda x: (x == 'positive').sum() - (x == 'negative').sum()
                ).reset_index()
                
                fig3 = px.line(daily_sentiment, x='date', y='sentiment', title="Daily Sentiment Balance", markers=True)
                fig3.update_layout(xaxis_title="Date", yaxis_title="Sentiment Score (Pos - Neg)")
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            if include_aspects:
                st.markdown("#### 🔍 Aspect-Based Analysis")
                all_aspects = []
                for aspects_list in results_df.get('aspects', []):
                    if aspects_list:
                        all_aspects.extend(aspects_list)
                
                if all_aspects:
                    aspects_df = pd.DataFrame(all_aspects)
                    aspect_counts = aspects_df['aspect'].value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        fig4 = px.bar(x=aspect_counts.index, y=aspect_counts.values, title="Most Discussed Aspects", labels={'x': 'Aspect', 'y': 'Count'}, color=aspect_counts.index)
                        st.plotly_chart(fig4, use_container_width=True)
                    with col2:
                        if 'sentiment' in aspects_df.columns:
                            aspect_sentiment = aspects_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
                            fig5 = px.bar(aspect_sentiment, title="Sentiment by Aspect", barmode='group')
                            st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.info("No aspects detected in the text with current criteria.")
        
        with tab4:
            if include_emotions:
                st.markdown("#### 😊 Emotion Analysis")
                if 'dominant_emotion' in results_df.columns:
                    emotion_counts = results_df['dominant_emotion'].value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        fig6 = px.bar(x=emotion_counts.index, y=emotion_counts.values, title="Dominant Emotions", labels={'x': 'Emotion', 'y': 'Count'}, color=emotion_counts.index, color_discrete_sequence=px.colors.qualitative.Set3)
                        st.plotly_chart(fig6, use_container_width=True)
                    with col2:
                        emotion_text = " ".join(results_df[results_df['dominant_emotion'] != 'neutral']['cleaned_text'])
                        if emotion_text:
                            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(emotion_text)
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title('Emotion-related Words')
                            st.pyplot(fig)
                
                st.markdown("#### 🔗 Emotion vs Sentiment")
                if 'dominant_emotion' in results_df.columns and 'sentiment' in results_df.columns:
                    emotion_sentiment = results_df.groupby(['dominant_emotion', 'sentiment']).size().unstack(fill_value=0)
                    fig7 = px.imshow(emotion_sentiment, title="Emotion-Sentiment Heatmap", color_continuous_scale='RdBu')
                    st.plotly_chart(fig7, use_container_width=True)
        
        with tab5:
            st.markdown("#### 📊 Advanced Analytics")
            col1, col2 = st.columns(2)
            with col1:
                if include_topics:
                    topics = extract_topics(results_df['cleaned_text'].tolist(), n_topics=3)
                    if topics:
                        st.markdown("##### 🗂️ Detected Topics")
                        for topic in topics:
                            with st.container():
                                st.markdown(f"**Topic {topic['topic_id']}**")
                                st.caption(f"Keywords: {', '.join(topic['keywords'])}")
                                st.progress(min(topic['frequency'] / 50, 1.0))
                                st.caption(f"Frequency: {topic['frequency']} mentions")
                                st.divider()
            with col2:
                if include_trends and 'date' in results_df.columns:
                    sorted_results = results_df.sort_values('date')
                    trend, forecast = forecast_sentiment_trend(sorted_results['sentiment'].tolist(), sorted_results['date'].tolist())
                    st.markdown("##### 📈 Trend Analysis")
                    if trend:
                        trend_icon = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
                        forecast_icon = "🟢" if forecast == "positive" else "🔴" if forecast == "negative" else "🟡"
                        st.metric("Current Trend", f"{trend_icon} {trend.title()}")
                        st.metric("Short-term Forecast", f"{forecast_icon} {forecast.title()}")
            
            st.markdown("#### ☁️ Word Clouds by Sentiment")
            sentiment_cols = st.columns(3)
            sentiments = ['positive', 'neutral', 'negative']
            for idx, sentiment in enumerate(sentiments):
                with sentiment_cols[idx]:
                    sentiment_texts = results_df[results_df['sentiment'] == sentiment]['cleaned_text']
                    if len(sentiment_texts) > 0:
                        text_combined = " ".join(sentiment_texts)
                        wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='viridis' if sentiment == 'positive' else 'cool' if sentiment == 'neutral' else 'Reds').generate(text_combined)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'{sentiment.title()} Words')
                        st.pyplot(fig)
        
        st.markdown('<div class="section-header"><h4 style="margin: 0;">📥 Export Results</h4></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download CSV", data=csv_data, file_name="sentiment_analysis_results.csv", mime="text/csv", use_container_width=True)
        with col2:
            json_data = results_df.to_json(orient='records', indent=2)
            st.download_button("📊 Download JSON", data=json_data.encode('utf-8'), file_name="sentiment_analysis_results.json", mime="application/json", use_container_width=True)
        with col3:
            summary = {
                "total_analyzed": len(results_df),
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "neutral_count": int(neutral_count),
                "avg_confidence": float(avg_confidence),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            summary_json = json.dumps(summary, indent=2)
            st.download_button("📋 Download Summary", data=summary_json.encode('utf-8'), file_name="analysis_summary.json", mime="application/json", use_container_width=True)
        
        if st.button("🔄 Clear Analysis & Reset", use_container_width=True):
            st.session_state.run_analysis = False
            st.session_state.analysis_completed = False
            st.rerun()
    else:
        st.markdown("### 📋 Data Preview")
        st.dataframe(data.head(), use_container_width=True)
        st.info("👈 Click **'Start Analysis'** in the sidebar to begin processing text data.")
else:
    st.warning("Please select or upload a valid dataset first.")

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">
<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <p style="margin: 0.5rem 0; color: #4b5563;">Advanced sentiment analysis for comprehensive text understanding</p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">© 2026 Advanced Sentiment Analyzer Pro</p>
</div>
""", unsafe_allow_html=True)