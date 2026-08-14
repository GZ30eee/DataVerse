import streamlit as st
import pandas as pd
import numpy as np
import random
import requests
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 🔥 API Keys (Loaded from secrets.toml)
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")
TMDB_API_KEY = st.secrets.get("tmdb", {}).get("api_key")

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Custom JSON encoder to handle numpy/pandas types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, datetime):
            return obj.isoformat()    
        else:
            return super(NumpyEncoder, self).default(obj)

# Set page config
st.set_page_config(page_title="🎬 Advanced Movie Recommender Pro", layout="wide")

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
    .movie-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #10b981;
        transition: transform 0.3s ease;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    .section-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
    }
    .recommendation-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎬 Advanced Movie Recommender Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        AI-Powered Movie Discovery with Vector Similarity, Collaborative Filtering & Hybrid Approaches
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for user interactions
if 'user_session' not in st.session_state:
    st.session_state.user_session = {
        'session_id': str(random.randint(100000, 999999)),
        'start_time': datetime.now(),
        'viewed_movies': [],
        'liked_movies': [],
        'searched_genres': [],
        'recommendation_history': []
    }

if 'movie_embeddings' not in st.session_state:
    st.session_state.movie_embeddings = None

if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None

if 'movie_vectors' not in st.session_state:
    st.session_state.movie_vectors = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 🎬 Recommendation Mode")
    recommendation_mode = st.selectbox(
        "Choose Mode:",
        ["Content-Based Filtering", "Collaborative Filtering", "Hybrid Approach", "Cold Start"]
    )
    
    # Advanced Features Toggles
    st.markdown("---")
    st.markdown("#### 🔧 Advanced Features")
    
    include_vector_similarity = st.checkbox("🔍 Vector Similarity Search", value=True)
    include_multi_modal = st.checkbox("🖼️ Multi-modal Retrieval", value=False)
    include_session_tracking = st.checkbox("📊 Session-based Recommendations", value=True)
    include_cold_start = st.checkbox("❄️ Cold-start Handling", value=True)
    include_explanations = st.checkbox("🤖 AI Explanations", value=True)
    
    # OpenAI Status
    st.markdown("---")
    st.markdown("#### 🤖 AI Status")
    if client:
        st.success("✅ OpenAI Connected")
        st.caption("AI explanations enabled")
    else:
        st.warning("⚠️ OpenAI Not Connected")
        st.caption("Add API key in .streamlit/secrets.toml")
    
    st.markdown("---")
    st.markdown("#### 📊 Session Info")
    st.caption(f"Session ID: {st.session_state.user_session['session_id']}")
    st.caption(f"Movies Viewed: {len(st.session_state.user_session['viewed_movies'])}")
    st.caption(f"Movies Liked: {len(st.session_state.user_session['liked_movies'])}")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.user_session = {
            'session_id': str(random.randint(100000, 999999)),
            'start_time': datetime.now(),
            'viewed_movies': [],
            'liked_movies': [],
            'searched_genres': [],
            'recommendation_history': []
        }
        st.rerun()

# Load and prepare movie dataset with caching
@st.cache_data
def load_movie_data():
    """Load and preprocess movie dataset with caching"""
    try:
        movies = pd.read_csv("dataset/movies.csv")
    except:
        sample_data = {
            'movieId': range(1, 101),
            'title': [f"Movie {i}" for i in range(1, 101)],
            'genres': ["Action|Adventure|Sci-Fi" if i % 3 == 0 else 
                      "Comedy|Romance" if i % 3 == 1 else 
                      "Drama|Thriller" for i in range(1, 101)],
            'year': [random.randint(1990, 2023) for _ in range(100)],
            'rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(100)],
            'votes': [random.randint(1000, 100000) for _ in range(100)]
        }
        movies = pd.DataFrame(sample_data)
    
    movies['content_features'] = movies['title'] + " " + movies['genres'].str.replace('|', ' ')
    
    if 'year' not in movies.columns:
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(2000).astype(int)
    
    if 'rating' not in movies.columns:
        movies['rating'] = np.round(np.random.uniform(3.0, 5.0, len(movies)), 1)
    
    if 'votes' not in movies.columns:
        movies['votes'] = np.random.randint(1000, 100000, len(movies))
    
    return movies

movies = load_movie_data()

# Vector Similarity & Matrix Functions
@st.cache_data
def create_movie_embeddings(movies_df_hashable_proxy):
    """Create TF-IDF embeddings for movie content with caching"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(movies['content_features'])
    
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(tfidf_matrix.toarray())
    
    return tfidf_matrix, vectors_2d

def get_vector_similarity(movie_title, movies_df, tfidf_matrix, n_recommendations=5):
    """Find similar movies using vector similarity"""
    if movie_title not in movies_df['title'].values:
        return pd.DataFrame()
    
    movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
    movie_vector = tfidf_matrix[movie_idx]
    
    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
    
    return movies_df.iloc[similar_indices]

# Collaborative Filtering Implementation
def get_collaborative_recommendations(user_session, movies_df, n_recommendations=5):
    """Implement item/rating-based collaborative filtering recommendations"""
    if not user_session['liked_movies']:
        return get_cold_start_recommendations(movies_df, n_recommendations)
    
    liked_genres = []
    for title in user_session['liked_movies']:
        match = movies_df[movies_df['title'] == title]
        if not match.empty:
            genres = match['genres'].iloc[0].split('|')
            liked_genres.extend(genres)
            
    # Score items based on rating, votes, and overlap with liked user genres
    def collaborative_score(row):
        score = row['rating'] * np.log1p(row['votes'])
        row_genres = str(row['genres']).split('|')
        overlap = len(set(row_genres).intersection(set(liked_genres)))
        return score * (1.0 + (overlap * 0.5))

    scored_df = movies_df.copy()
    scored_df['collab_score'] = scored_df.apply(collaborative_score, axis=1)
    
    # Exclude already liked or viewed movies
    excluded = set(user_session['liked_movies'] + user_session['viewed_movies'])
    scored_df = scored_df[~scored_df['title'].isin(excluded)]
    
    return scored_df.nlargest(n_recommendations, 'collab_score')

# Hybrid Approach Implementation
def get_hybrid_recommendations(user_session, movie_title, movies_df, tfidf_matrix, n_recommendations=5):
    """Combine content-based similarity and collaborative scoring weights into a hybrid score"""
    if movie_title not in movies_df['title'].values:
        return get_cold_start_recommendations(movies_df, n_recommendations)
        
    movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
    movie_vector = tfidf_matrix[movie_idx]
    content_sims = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    hybrid_scores = []
    for idx, row in movies_df.iterrows():
        c_sim = content_sims[idx]
        pop_score = (row['rating'] / 5.0) * (np.log1p(row['votes']) / 12.0)
        # Hybrid formula: 60% content similarity + 40% collaborative popularity score
        h_score = (0.6 * c_sim) + (0.4 * pop_score)
        hybrid_scores.append((idx, h_score))
        
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in hybrid_scores if idx != movie_idx][:n_recommendations]
    
    return movies_df.iloc[top_indices]

def get_movie_poster(movie_title, year=None):
    """Get movie poster from TMDB API"""
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY" or not TMDB_API_KEY:
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {'api_key': TMDB_API_KEY, 'query': movie_title, 'year': year}
        response = requests.get(search_url, params=params, timeout=5)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                poster_path = results[0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None

def get_session_based_recommendations(user_session, movies_df, tfidf_matrix, n_recommendations=5):
    """Generate recommendations based on user's session history"""
    if not user_session['liked_movies']:
        return get_cold_start_recommendations(movies_df, n_recommendations)
    
    liked_indices = [movies_df[movies_df['title'] == t].index[0] for t in user_session['liked_movies'][-5:] if t in movies_df['title'].values]
    if not liked_indices:
        return get_cold_start_recommendations(movies_df, n_recommendations)
    
    liked_vectors = tfidf_matrix[liked_indices].mean(axis=0)
    similarities = cosine_similarity(liked_vectors, tfidf_matrix).flatten()
    
    viewed_indices = [movies_df[movies_df['title'] == t].index[0] for t in user_session['viewed_movies'] if t in movies_df['title'].values]
    if viewed_indices:
        similarities[viewed_indices] = -1
        
    recommended_indices = similarities.argsort()[::-1][:n_recommendations]
    return movies_df.iloc[recommended_indices]

def get_cold_start_recommendations(movies_df, n_recommendations=5):
    """Generate recommendations for new users (cold start)"""
    if 'rating' in movies_df.columns and 'votes' in movies_df.columns:
        movies_df['popularity_score'] = movies_df['rating'] * np.log1p(movies_df['votes'])
        recommended = movies_df.nlargest(n_recommendations, 'popularity_score')
    else:
        recommended = movies_df.sample(min(n_recommendations, len(movies_df)))
    return recommended

def generate_ai_explanation(client, movie_title, similar_movies, user_session=None):
    """Generate AI explanation for recommendations"""
    if not client:
        return None
    try:
        similar_titles = ", ".join(similar_movies['title'].head(3).tolist())
        context = f"The user recently liked: {', '.join(user_session['liked_movies'][-3:])}" if (user_session and user_session['liked_movies']) else "New user."
        prompt = f"Explaining movie recommendation for '{movie_title}'. Context: {context}. Recommendations: {similar_titles}. Provide a brief 2-3 sentence explanation of why they share themes."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a movie expert providing insightful recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

# Initialize vector embeddings with caching proxy
if st.session_state.movie_embeddings is None and include_vector_similarity:
    with st.spinner("🔍 Creating movie embeddings..."):
        tfidf_matrix, vectors_2d = create_movie_embeddings("cached_dataset_token")
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.movie_vectors = vectors_2d
        st.session_state.movie_embeddings = True

# Tabs setup
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Search & Discover", 
    "🔍 Similar Movies", 
    "📊 Your Session",
    "📈 Visualization",
    "⚙️ Settings"
])

with tab1:
    st.markdown('<div class="section-header"><h3 style="margin: 0;">🎯 Search & Discover Movies</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### 🎭 Browse by Genre")
        all_genres = set()
        for genres in movies['genres'].dropna():
            all_genres.update(genres.split('|'))
        
        selected_genres = st.multiselect(
            "Select genres:",
            sorted(all_genres),
            max_selections=3
        )
        
        if selected_genres:
            st.session_state.user_session['searched_genres'].extend(selected_genres)
            mask = pd.Series(False, index=movies.index)
            for genre in selected_genres:
                mask = mask | movies['genres'].str.contains(genre, case=False, na=False)
            
            filtered_movies = movies[mask]
            if not filtered_movies.empty:
                if 'rating' in filtered_movies.columns and 'votes' in filtered_movies.columns:
                    filtered_movies['score'] = filtered_movies['rating'] * np.log1p(filtered_movies['votes'])
                    filtered_movies = filtered_movies.sort_values('score', ascending=False)
                
                st.success(f"Found {len(filtered_movies)} movies matching your genres")
                for idx, movie in filtered_movies.head(10).iterrows():
                    with st.container():
                        col_a, col_b, col_c = st.columns([3, 1, 1])
                        with col_a:
                            st.markdown(f"**{movie['title']}**")
                            st.caption(f"🎭 {movie['genres']} | ⭐ {movie.get('rating', 'N/A')}/5.0")
                        with col_b:
                            if st.button("👍 Like", key=f"like_{idx}", use_container_width=True):
                                if movie['title'] not in st.session_state.user_session['liked_movies']:
                                    st.session_state.user_session['liked_movies'].append(movie['title'])
                                    st.success(f"Liked {movie['title']}!")
                        with col_c:
                            if st.button("👀 View", key=f"view_{idx}", use_container_width=True):
                                if movie['title'] not in st.session_state.user_session['viewed_movies']:
                                    st.session_state.user_session['viewed_movies'].append(movie['title'])
                                    st.rerun()
                        st.divider()
    
    with col2:
        st.markdown("#### 🎲 Quick Picks")
        if st.button("🎬 Popular Movies", use_container_width=True):
            st.session_state.quick_recommendations = movies.nlargest(5, 'votes')
            st.rerun()
        if st.button("⭐ Highly Rated", use_container_width=True):
            st.session_state.quick_recommendations = movies.nlargest(5, 'rating')
            st.rerun()
        
        if 'quick_recommendations' in st.session_state:
            st.markdown("##### Recommended:")
            for _, movie in st.session_state.quick_recommendations.iterrows():
                st.caption(f"• {movie['title']}")

with tab2:
    st.markdown('<div class="section-header"><h3 style="margin: 0;">🔍 Find Similar Movies</h3></div>', unsafe_allow_html=True)
    movie_search = st.text_input("🔍 Enter a movie title to find similar movies:")
    
    if movie_search:
        matching_movies = movies[movies['title'].str.contains(movie_search, case=False, na=False)]
        if not matching_movies.empty:
            selected_movie = st.selectbox("Select a movie:", matching_movies['title'].tolist())
            if selected_movie:
                if selected_movie not in st.session_state.user_session['viewed_movies']:
                    st.session_state.user_session['viewed_movies'].append(selected_movie)
                
                # Routing recommendation logic based on sidebar selection
                if recommendation_mode == "Collaborative Filtering":
                    similar_movies = get_collaborative_recommendations(st.session_state.user_session, movies, n_recommendations=10)
                elif recommendation_mode == "Hybrid Approach" and st.session_state.tfidf_matrix is not None:
                    similar_movies = get_hybrid_recommendations(st.session_state.user_session, selected_movie, movies, st.session_state.tfidf_matrix, n_recommendations=10)
                elif recommendation_mode == "Cold Start":
                    similar_movies = get_cold_start_recommendations(movies, n_recommendations=10)
                else:  # Content-Based Filtering
                    if include_vector_similarity and st.session_state.tfidf_matrix is not None:
                        similar_movies = get_vector_similarity(selected_movie, movies, st.session_state.tfidf_matrix, n_recommendations=10)
                    else:
                        similar_movies = get_cold_start_recommendations(movies, n_recommendations=10)
                
                if not similar_movies.empty:
                    st.success(f"Generated {len(similar_movies)} recommendations using mode: **{recommendation_mode}**")
                    col_a, col_b = st.columns([3, 2])
                    with col_a:
                        for idx, movie in similar_movies.head(5).iterrows():
                            with st.container():
                                st.markdown(f"**{movie['title']}**")
                                st.caption(f"🎭 {movie['genres']} | ⭐ {movie.get('rating', 'N/A')}/5.0")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("👍 Like", key=f"sim_like_{idx}", use_container_width=True):
                                        if movie['title'] not in st.session_state.user_session['liked_movies']:
                                            st.session_state.user_session['liked_movies'].append(movie['title'])
                                            st.success(f"Liked {movie['title']}!")
                                with col2:
                                    if st.button("👀 View", key=f"sim_view_{idx}", use_container_width=True):
                                        if movie['title'] not in st.session_state.user_session['viewed_movies']:
                                            st.session_state.user_session['viewed_movies'].append(movie['title'])
                                            st.rerun()
                                st.divider()
                    with col_b:
                        if include_explanations and client:
                            if st.button("🤖 Get AI Explanation", use_container_width=True):
                                with st.spinner("Generating AI explanation..."):
                                    explanation = generate_ai_explanation(client, selected_movie, similar_movies, st.session_state.user_session)
                                    if explanation:
                                        st.info("**AI Explanation:**")
                                        st.write(explanation)

with tab3:
    st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Your Recommendation Session</h3></div>', unsafe_allow_html=True)
    if include_session_tracking:
        col1, col2, col3 = st.columns(3)
        with col1:
            session_duration = datetime.now() - st.session_state.user_session['start_time']
            st.metric("Session Duration", f"{session_duration.seconds//60} min")
        with col2:
            st.metric("Movies Viewed", len(st.session_state.user_session['viewed_movies']))
        with col3:
            st.metric("Movies Liked", len(st.session_state.user_session['liked_movies']))
        
        st.markdown("#### 🎯 Personalized Recommendations")
        if st.button("🎬 Get Personalized Recommendations", use_container_width=True):
            if st.session_state.tfidf_matrix is not None:
                personalized_recs = get_session_based_recommendations(st.session_state.user_session, movies, st.session_state.tfidf_matrix, n_recommendations=6)
                if not personalized_recs.empty:
                    for idx, (_, movie) in enumerate(personalized_recs.iterrows()):
                        st.markdown(f"• **{movie['title']}** ({movie['genres']})")

with tab4:
    st.markdown('<div class="section-header"><h3 style="margin: 0;">📈 Movie Visualization</h3></div>', unsafe_allow_html=True)
    if st.session_state.movie_vectors is not None:
        vectors_df = pd.DataFrame(st.session_state.movie_vectors, columns=['x', 'y'])
        vectors_df['title'] = movies['title'].values[:len(vectors_df)]
        vectors_df['genres'] = movies['genres'].values[:len(vectors_df)]
        vectors_df['primary_genre'] = vectors_df['genres'].apply(lambda x: str(x).split('|')[0] if pd.notna(x) else 'Unknown')
        
        fig = px.scatter(vectors_df.head(100), x='x', y='y', color='primary_genre', hover_data=['title', 'genres'], title='Movie Vector Space (PCA Reduced)')
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown('<div class="section-header"><h3 style="margin: 0;">⚙️ Settings & Export</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Movies", len(movies))
    with col2:
        session_json = json.dumps(st.session_state.user_session, indent=2, cls=NumpyEncoder)
        st.download_button("📋 Export Session Data", data=session_json.encode('utf-8'), file_name=f"movie_session_{st.session_state.user_session['session_id']}.json", mime="application/json", use_container_width=True)

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">
<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <p>© 2026 Advanced Movie Recommender Pro[cite: 3]</p>
</div>
""", unsafe_allow_html=True)