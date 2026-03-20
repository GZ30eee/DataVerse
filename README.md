# 🌌 DataVerse: Advanced Analytical Suite

**DataVerse** is a comprehensive collection of AI-powered analysis applications and machine learning tools, all unified under a single, professional interface. 🚀

---

## 🚀 Key Applications

### 🧹 Cleaning Assistant Pro
AI-powered data cleaning with smart imputation, anomaly detection, and privacy protection. 
- **Features**: PII preservation, LLM-based anomaly detection, schema inference, and quality reporting.
- **File**: `cleaning_assisstant.py`

### 📈 Stock Analyzer
Multi-indicator financial analysis with real-time data and prediction models.
- **Features**: TA indicators (RSI, MACD, Bollinger), Sentiment analysis from news, and Random Forest price prediction.
- **File**: `stock.py`

### 🎵 Music Analyzer Pro
Deep audio analysis using advanced feature extraction and machine learning.
- **Features**: Genre classification (MFCC), Cross-modal similarity, Playlist generation, and Style Transfer suggestions.
- **File**: `music.py`

### 🎭 Sentiment Analyzer Pro
Advanced text analysis beyond basic polarity.
- **Features**: Aspect extraction, Emotion detection (Joy, Anger, Trust, etc.), Topic modeling, and Trend forecasting.
- **File**: `sen.py`

### 🌳 Decision Tree Explorer
Interpretable machine learning visualization and analysis.
- **Features**: SHAP value analysis, Counterfactual explanations, Tree pruning, and Model Distillation.
- **File**: `decision_tree.py`

### 🧬 AI Cluster Vision
Multi-algorithm clustering visualization.
- **Features**: KMeans, DBSCAN, Agglomerative, Elbow method analysis, and 3D space visualization.
- **File**: `cluster.py`

### 🛒 Association Rules (Market Basket)
Discover relationships between variables in large datasets.
- **Features**: Frequency itemset discovery, AI-generated rule insights, and network relation graphs.
- **File**: `association_rules.py`

### 🎥 Movie Recommendation System
Personalized content discovery engine.
- **Features**: TF-IDF similarity, Genre filtering, and TMDB integration.
- **File**: `movie.py`

---

## 🛠️ Technology Stack
- **Core Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-Learn, SHAP, UMAP, XGBoost
- **Visualization**: Plotly, Seaborn, Matplotlib, NetworkX
- **Audio Processing**: Librosa, Soundfile
- **Natural Language Processing**: TextBlob, WordCloud, OpenAI GPT-4o
- **Finance**: yfinance, ta (Technical Analysis Library)

---

## 📥 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GZ30eee/DataVerse.git
   cd DataVerse
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Secrets**:
   Create a `.streamlit/secrets.toml` file with your API keys:
   ```toml
   [openai]
   api_key = "YOUR_OPENAI_KEY"
   
   [newsapi]
   api_key = "YOUR_NEWS_API_KEY"
   ```

---

## 🚀 How to Run
Launch the unified dashboard:
```bash
# Open the main portal (HTML based)
main.html
```

Or run any specific application directly:
```bash
streamlit run <app_name>.py
```

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` (if any) for more information.

Developed by **Ghanshyamsinh Zala** 👨‍💻
