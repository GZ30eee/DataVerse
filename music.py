import joblib
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import io
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load trained model and label encoder
model = joblib.load('models/music_genre_classifier.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Set up Streamlit App
st.set_page_config(page_title="🎵 Advanced Music Analyzer Pro", layout="wide")

# Professional CSS (similar to sentiment analyzer)
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
    .genre-card {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .genre-card:hover {
        transform: translateY(-2px);
    }
    .audio-player {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎵 Advanced Music Analyzer Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Multi-modal Music Analysis with AI Classification, Cross-modal Search & Playlist Generation
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 🎵 Analysis Mode")
    analysis_mode = st.radio(
        "Choose analysis mode:",
        ["🎵 Single Track Analysis", "📁 Batch Processing", "🔍 Cross-modal Search"],
        index=0
    )
    
    # Advanced features
    st.markdown("---")
    st.markdown("#### 🔧 Advanced Features")
    
    include_embeddings = st.checkbox("🧠 Audio Feature Embeddings", value=True)
    include_cross_modal = st.checkbox("🔄 Cross-modal Search", value=True)
    include_playlist = st.checkbox("📋 Playlist Generation", value=True)
    include_similarity = st.checkbox("🔗 Similarity Graph", value=True)
    include_style_transfer = st.checkbox("🎨 Style Transfer Suggestions", value=True)
    
    st.markdown("---")
    st.markdown("#### 🎯 Analysis Depth")
    
    analysis_depth = st.select_slider(
        "Analysis Depth:",
        options=["Basic", "Standard", "Advanced", "Expert"],
        value="Advanced"
    )
    
    # Confidence threshold
    min_confidence = st.slider("Minimum Confidence", 0.5, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

# Initialize session state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'audio_features' not in st.session_state:
    st.session_state.audio_features = {}
if 'similarity_graph' not in st.session_state:
    st.session_state.similarity_graph = None

# Sample audio files for testing
sample_files = {
    "Rock Sample": "samples/rock.mp3",
    "Jazz Sample": "samples/jazz.mp3",
    "Hip Hop Sample": "samples/hiphop.mp3",
    "Classical Sample": "samples/classical.mp3",
    "Pop Sample": "samples/pop.mp3",
    "Blues Sample": "samples/blues.mp3",
    "Country Sample": "samples/country.mp3",
    "Electronic Sample": "samples/electronic.mp3",
    "Reggae Sample": "samples/reggae.mp3",
    "Metal Sample": "samples/metal.mp3"
}

# Function to extract MFCC features
def extract_mfcc_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Enhanced feature extraction (simulating VGGish/CLAP-like embeddings)
def extract_audio_embeddings(audio_data, sr):
    # This simulates more sophisticated embeddings
    # In production, you'd use actual VGGish or CLAP models
    
    # Extract multiple features
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    features['spectral_centroid'] = np.mean(spectral_centroid)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)
    features['zcr'] = np.mean(zcr)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    features['tempo'] = tempo[0] if len(tempo) > 0 else 120
    
    # Combine all features
    all_features = []
    for key in ['mfcc_mean', 'mfcc_std', 'chroma_mean']:
        all_features.extend(features[key])
    all_features.extend([features['spectral_centroid'], features['zcr'], features['tempo']])
    
    return np.array(all_features), features

# Cross-modal search function
def cross_modal_search(query_text, audio_embeddings, metadata):
    """Search audio by text description"""
    # In production, use CLAP or similar models for actual cross-modal embedding
    # This is a simplified version
    query_embeddings = {
        "relaxing": [0.8, 0.1, 0.1],
        "energetic": [0.1, 0.8, 0.1],
        "melodic": [0.3, 0.3, 0.4],
        "rhythmic": [0.2, 0.6, 0.2],
        "emotional": [0.6, 0.2, 0.2]
    }
    
    # Get query embedding (simplified)
    query_vec = query_embeddings.get(query_text.lower(), [0.5, 0.5, 0.5])
    
    # Calculate similarities
    similarities = []
    for idx, emb in enumerate(audio_embeddings):
        # Simplified similarity calculation
        if len(emb) >= 3:
            sim = cosine_similarity([query_vec], [emb[:3]])[0][0]
            similarities.append((idx, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:10]

# Playlist generation
def generate_playlist(seed_track_idx, audio_embeddings, metadata, n_tracks=10):
    """Generate playlist based on track similarity"""
    similarities = []
    seed_emb = audio_embeddings[seed_track_idx]
    
    for idx, emb in enumerate(audio_embeddings):
        if idx != seed_track_idx:
            sim = cosine_similarity([seed_emb], [emb])[0][0]
            similarities.append((idx, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Create playlist
    playlist = [seed_track_idx]
    playlist.extend([idx for idx, _ in similarities[:n_tracks-1]])
    
    return playlist

# Create similarity graph
def create_similarity_graph(audio_embeddings, metadata, threshold=0.7):
    """Create network graph of similar tracks"""
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(audio_embeddings)):
        G.add_node(i, 
                   label=metadata[i].get('title', f"Track {i+1}"),
                   genre=metadata[i].get('genre', 'Unknown'))
    
    # Add edges based on similarity
    for i in range(len(audio_embeddings)):
        for j in range(i+1, len(audio_embeddings)):
            sim = cosine_similarity([audio_embeddings[i]], [audio_embeddings[j]])[0][0]
            if sim > threshold:
                G.add_edge(i, j, weight=sim)
    
    return G

# Style transfer suggestions
def get_style_transfer_suggestions(track_features, genre_features_db):
    """Suggest style transfer options"""
    suggestions = []
    
    # Calculate distances to other genres
    for genre, genre_feats in genre_features_db.items():
        if len(track_features) == len(genre_feats):
            distance = np.linalg.norm(track_features - genre_feats)
            suggestions.append({
                'target_genre': genre,
                'similarity_score': 1.0 / (1.0 + distance),
                'suggested_changes': []
            })
    
    # Sort by similarity
    suggestions.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Add suggested changes
    for suggestion in suggestions[:3]:
        if suggestion['target_genre'] == 'Rock':
            suggestion['suggested_changes'] = ['Increase tempo by 15%', 'Add distortion to guitars']
        elif suggestion['target_genre'] == 'Jazz':
            suggestion['suggested_changes'] = ['Add swing rhythm', 'Include piano improvisation']
        elif suggestion['target_genre'] == 'Electronic':
            suggestion['suggested_changes'] = ['Add synthesizer pads', 'Increase BPM to 128']
    
    return suggestions[:3]

# Initialize database of genre features (simulated)
genre_features_db = {
    'Rock': np.random.randn(50),
    'Jazz': np.random.randn(50),
    'Hip Hop': np.random.randn(50),
    'Classical': np.random.randn(50),
    'Pop': np.random.randn(50),
    'Blues': np.random.randn(50),
    'Country': np.random.randn(50),
    'Electronic': np.random.randn(50),
    'Reggae': np.random.randn(50),
    'Metal': np.random.randn(50)
}

# Main UI
if analysis_mode == "🎵 Single Track Analysis":
    st.markdown("### 🎵 Single Track Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'm4a', 'flac'])
    
    with col2:
        selected_sample = st.selectbox("Or try a sample:", ["None"] + list(sample_files.keys()))
    
    # Load audio
    audio_data = None
    sr = None
    metadata = {}
    
    if selected_sample != "None" and selected_sample in sample_files:
        # In production, load actual file
        # For demo, create synthetic features
        sr = 22050
        audio_data = np.random.randn(sr * 30)  # 30 seconds of audio
        metadata = {
            'title': selected_sample,
            'duration': 30,
            'sample_rate': sr
        }
        st.audio(np.random.randn(sr * 5), sample_rate=sr)  # Preview
    
    elif uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        metadata = {
            'title': uploaded_file.name,
            'duration': len(audio_data) / sr,
            'sample_rate': sr
        }
        st.audio(uploaded_file, format='audio/wav')
    
    if audio_data is not None and st.session_state.run_analysis:
        with st.spinner("🔍 Analyzing audio features..."):
            
            # Extract features
            mfcc_features = extract_mfcc_features(audio_data, sr)
            
            if include_embeddings:
                embeddings, detailed_features = extract_audio_embeddings(audio_data, sr)
            
            # Predict genre
            prediction = model.predict(mfcc_features.reshape(1, -1))
            predicted_genre = label_encoder.inverse_transform(prediction)[0]
            confidence = np.max(np.abs(model.decision_function(mfcc_features.reshape(1, -1))))
            
            # Store in session state
            track_id = len(st.session_state.audio_features)
            st.session_state.audio_features[track_id] = {
                'embeddings': embeddings if include_embeddings else mfcc_features,
                'metadata': {**metadata, 'genre': predicted_genre},
                'detailed_features': detailed_features if include_embeddings else None
            }
            
            # Display Results
            st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Analysis Results</h3></div>', unsafe_allow_html=True)
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎶 Predicted Genre", predicted_genre)
            with col2:
                st.metric("📊 Confidence", f"{confidence:.1%}")
            with col3:
                duration_min = metadata['duration'] / 60
                st.metric("⏱️ Duration", f"{duration_min:.1f} min")
            with col4:
                st.metric("🎵 Sample Rate", f"{sr/1000:.1f} kHz")
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎵 Audio Features", 
                "📈 Visualizations", 
                "🔍 Cross-modal Search",
                "📋 Playlist Generation",
                "🎨 Style Transfer"
            ])
            
            with tab1:
                # Display audio features
                st.markdown("#### 🧠 Audio Features")
                
                if include_embeddings:
                    # Feature cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("##### 📊 MFCC Features")
                        st.write(f"Mean: {np.mean(detailed_features['mfcc_mean']):.3f}")
                        st.write(f"Std: {np.mean(detailed_features['mfcc_std']):.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("##### 🎼 Chroma Features")
                        st.write(f"Mean Chroma: {np.mean(detailed_features['chroma_mean']):.3f}")
                        st.progress(float(np.mean(detailed_features['chroma_mean'])))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("##### ⚡ Rhythmic Features")
                        st.write(f"Tempo: {detailed_features['tempo']:.1f} BPM")
                        st.write(f"Spectral Centroid: {detailed_features['spectral_centroid']:.1f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature visualization
                st.markdown("#### 📈 Feature Distribution")
                if include_embeddings:
                    feature_df = pd.DataFrame({
                        'MFCC': detailed_features['mfcc_mean'][:10],
                        'Chroma': detailed_features['chroma_mean'][:10]
                    })
                    st.line_chart(feature_df)
            
            with tab2:
                # Audio visualizations
                st.markdown("#### 📊 Audio Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Waveform
                    fig, ax = plt.subplots(figsize=(8, 3))
                    librosa.display.waveshow(audio_data[:sr*5], sr=sr, ax=ax, alpha=0.7)
                    ax.set_title("Waveform (First 5 seconds)")
                    st.pyplot(fig)
                
                with col2:
                    # Spectrogram
                    fig, ax = plt.subplots(figsize=(8, 3))
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data[:sr*5])), ref=np.max)
                    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
                    ax.set_title("Spectrogram")
                    plt.colorbar(img, ax=ax, format='%+2.0f dB')
                    st.pyplot(fig)
                
                # Additional visualizations
                if include_embeddings:
                    # Tempo histogram
                    st.markdown("#### 🥁 Rhythm Analysis")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
                    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
                    ax.hist(onset_env, bins=50, alpha=0.7)
                    ax.set_xlabel("Onset Strength")
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Detected Tempo: {tempo[0]:.1f} BPM")
                    st.pyplot(fig)
            
            with tab3:
                if include_cross_modal:
                    st.markdown("#### 🔍 Cross-modal Search")
                    
                    # Text search input
                    search_query = st.text_input("Describe the music you're looking for:", "energetic rock")
                    
                    if st.button("🔎 Search Similar"):
                        # Get all embeddings from session state
                        all_embeddings = []
                        all_metadata = []
                        
                        for track_id, data in st.session_state.audio_features.items():
                            all_embeddings.append(data['embeddings'])
                            all_metadata.append(data['metadata'])
                        
                        # Add some sample embeddings for demo
                        for _ in range(5):
                            all_embeddings.append(np.random.randn(50))
                            all_metadata.append({
                                'title': f"Sample Track {_}",
                                'genre': np.random.choice(list(genre_features_db.keys()))
                            })
                        
                        # Perform search
                        results = cross_modal_search(search_query, all_embeddings, all_metadata)
                        
                        # Display results
                        st.markdown(f"**Found {len(results)} similar tracks:**")
                        for idx, (track_idx, similarity) in enumerate(results[:5]):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    metadata = all_metadata[track_idx]
                                    st.write(f"**{idx+1}. {metadata.get('title', f'Track {track_idx+1}')}**")
                                    st.write(f"Genre: {metadata.get('genre', 'Unknown')}")
                                with col2:
                                    st.metric("Similarity", f"{similarity:.2%}")
                                st.divider()
            
            with tab4:
                if include_playlist and len(st.session_state.audio_features) > 1:
                    st.markdown("#### 📋 Playlist Generation")
                    
                    # Select seed track
                    track_options = {f"Track {i+1} - {data['metadata'].get('genre', 'Unknown')}": i 
                                    for i, data in st.session_state.audio_features.items()}
                    
                    seed_track = st.selectbox("Select a seed track:", list(track_options.keys()))
                    
                    if st.button("🎵 Generate Playlist"):
                        seed_idx = track_options[seed_track]
                        
                        # Get all embeddings
                        all_embeddings = [data['embeddings'] for data in st.session_state.audio_features.values()]
                        all_metadata = [data['metadata'] for data in st.session_state.audio_features.values()]
                        
                        # Generate playlist
                        playlist_indices = generate_playlist(seed_idx, all_embeddings, all_metadata)
                        
                        # Display playlist
                        st.markdown("### 🎶 Generated Playlist")
                        for i, track_idx in enumerate(playlist_indices):
                            metadata = all_metadata[track_idx]
                            col1, col2, col3 = st.columns([1, 3, 1])
                            with col1:
                                st.write(f"**{i+1}.**")
                            with col2:
                                st.write(f"**{metadata.get('title', f'Track {track_idx+1}')}**")
                                st.write(f"Genre: {metadata.get('genre', 'Unknown')}")
                            with col3:
                                st.button("▶️", key=f"play_{track_idx}")
                            st.divider()
            
            with tab5:
                if include_style_transfer:
                    st.markdown("#### 🎨 Style Transfer Suggestions")
                    
                    # Get current track features
                    current_features = embeddings if include_embeddings else mfcc_features
                    
                    # Get suggestions
                    suggestions = get_style_transfer_suggestions(current_features, genre_features_db)
                    
                    # Display suggestions
                    for suggestion in suggestions:
                        with st.container():
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric(
                                    "Target Genre", 
                                    suggestion['target_genre'],
                                    delta=f"{suggestion['similarity_score']:.1%} match"
                                )
                            with col2:
                                st.markdown("**Suggested Changes:**")
                                for change in suggestion['suggested_changes']:
                                    st.write(f"• {change}")
                            st.divider()
            
            # Export section
            st.markdown('<div class="section-header"><h4 style="margin: 0;">📥 Export Results</h4></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export features as CSV
                if include_embeddings:
                    features_df = pd.DataFrame({
                        'feature': list(detailed_features.keys()),
                        'value': [np.mean(v) if isinstance(v, np.ndarray) else v for v in detailed_features.values()]
                    })
                    csv_data = features_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Export Features (CSV)",
                        data=csv_data,
                        file_name="audio_features.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                # Export analysis report
                report = {
                    "track_name": metadata.get('title', 'Unknown'),
                    "predicted_genre": predicted_genre,
                    "confidence": float(confidence),
                    "duration": float(metadata['duration']),
                    "sample_rate": int(sr),
                    "analysis_date": datetime.now().isoformat(),
                    "features_extracted": include_embeddings
                }
                import json
                json_data = json.dumps(report, indent=2)
                st.download_button(
                    label="📋 Export Report (JSON)",
                    data=json_data.encode('utf-8'),
                    file_name="audio_analysis_report.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Save playlist
                if include_playlist and 'playlist_indices' in locals():
                    playlist_data = {
                        "seed_track": seed_track,
                        "tracks": [
                            {
                                "id": idx,
                                "metadata": all_metadata[idx]
                            }
                            for idx in playlist_indices
                        ]
                    }
                    playlist_json = json.dumps(playlist_data, indent=2)
                    st.download_button(
                        label="🎵 Export Playlist",
                        data=playlist_json.encode('utf-8'),
                        file_name="generated_playlist.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.run_analysis = False
                st.rerun()

elif analysis_mode == "📁 Batch Processing":
    st.markdown("### 📁 Batch Processing")
    st.info("Batch processing feature coming soon!")
    
elif analysis_mode == "🔍 Cross-modal Search":
    st.markdown("### 🔍 Cross-modal Search")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Describe the music:", placeholder="e.g., relaxing piano music, energetic rock, emotional vocals")
    with col2:
        search_button = st.button("🔎 Search", use_container_width=True)
    
    if search_button and search_query:
        # Simulate search results
        results = [
            {"title": "Morning Meditation", "genre": "Classical", "similarity": 0.92},
            {"title": "Peaceful Piano", "genre": "Classical", "similarity": 0.88},
            {"title": "Ambient Dreams", "genre": "Electronic", "similarity": 0.85},
            {"title": "Calm Ocean Waves", "genre": "Ambient", "similarity": 0.82},
            {"title": "Zen Garden", "genre": "World", "similarity": 0.78},
        ]
        
        st.markdown(f"**Found {len(results)} results for '{search_query}':**")
        
        for result in results:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{result['title']}**")
                    st.write(f"Genre: {result['genre']}")
                with col2:
                    st.metric("Match", f"{result['similarity']:.0%}")
                with col3:
                    st.button("▶️ Play", key=f"play_{result['title']}")
                st.divider()

# Similarity Graph Visualization
if include_similarity and len(st.session_state.audio_features) > 2:
    st.markdown('<div class="section-header"><h3 style="margin: 0;">🔗 Similarity Graph</h3></div>', unsafe_allow_html=True)
    
    # Create graph
    all_embeddings = [data['embeddings'] for data in st.session_state.audio_features.values()]
    all_metadata = [data['metadata'] for data in st.session_state.audio_features.values()]
    
    G = create_similarity_graph(all_embeddings, all_metadata)
    
    # Visualize with plotly
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    genres = list(genre_features_db.keys())
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        metadata = all_metadata[node]
        node_text.append(f"{metadata.get('title', f'Track {node+1}')}<br>Genre: {metadata.get('genre', 'Unknown')}")
        
        # Color by genre
        genre = metadata.get('genre', 'Unknown')
        if genre in genres:
            node_color.append(genres.index(genre))
        else:
            node_color.append(len(genres))
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(i+1) for i in range(len(node_x))],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_color,
            colorscale='Viridis',
            line_width=2
        )
    ))
    
    fig.update_layout(
        title="Audio Similarity Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">

<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
        <span>🎵 <strong>Genre Classification</strong></span>
        <span>🧠 <strong>Audio Embeddings</strong></span>
        <span>🔄 <strong>Cross-modal Search</strong></span>
        <span>📋 <strong>Playlist Generation</strong></span>
        <span>🔗 <strong>Similarity Analysis</strong></span>
    </div>
    <p style="margin: 0.5rem 0; color: #4b5563;">
        Advanced multi-modal music analysis and recommendation system
    </p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">
        © 2024 Advanced Music Analyzer Pro
    </p>
</div>
""", unsafe_allow_html=True)