import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import mode
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Try importing openai with error handling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configuration constants dictionary
CONFIG = {
    "app_title": "🔍 Advanced K-Means Clustering",
    "layout": "wide",
    "default_n_init": 10,
    "max_iter": 300,
}

# Set up Streamlit App
st.set_page_config(page_title=CONFIG["app_title"], layout=CONFIG["layout"])

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
        border-left: 4px solid #10b981;
    }
    .section-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
    }
    .cluster-card {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🔍 Advanced K-Means Clustering Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Multi-dimensional Clustering with Auto-K Selection, Stability Testing & Sub-cluster Discovery
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    include_llm_labeling = st.checkbox("🤖 LLM Cluster Labeling", value=True)

    if include_llm_labeling:
        if not OPENAI_AVAILABLE:
            st.warning("⚠️ OpenAI package is not installed. LLM features will fall back to rule-based labeling.")
        else:
            st.info("LLM labeling active (reads from secrets.toml)")
        
    st.markdown("#### 📊 Dataset Generation")
    dataset_type = st.selectbox(
        "Dataset Type:",
        ["Blobs (Spherical Clusters)", "Moons (Non-linear)", "Circles (Concentric)", 
         "Anisotropic (Stretched)", "Noisy Circles", "Custom CSV Upload"]
    )
    
    n_samples = st.slider("Number of Samples", 100, 5000, 1000, 100, help="Total number of data points generated.")
    n_features = st.slider("Number of Features", 2, 20, 5, help="Dimensionality of the feature space.")
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.05, 0.05, help="Standard deviation or dispersion factor of noise.")
    random_state = st.slider("Random Seed", 0, 100, 42, help="Seed for reproducibility.")
    
    st.markdown("---")
    st.markdown("#### 🔧 Clustering Settings")

    clustering_method = st.radio(
        "K Selection Method:",
        ["Manual K", "Auto-K (Elbow + Silhouette)", "Auto-K (All Metrics)"]
    )

    if clustering_method == "Manual K":
        n_clusters = st.slider("Number of Clusters (K)", 2, 20, 5)
        max_clusters = 10 
    else:
        max_clusters = st.slider("Max Clusters to Test", 3, 20, 10)
    
    st.markdown("---")
    st.markdown("#### 🎯 Advanced Features")
    
    include_dimensionality = st.checkbox("📐 Multi-dimensional Visualization", value=True)
    include_stability = st.checkbox("⚖️ Cluster Stability Testing", value=True)
    include_subclusters = st.checkbox("🔍 Sub-cluster Discovery", value=True)
    include_metrics = st.checkbox("📊 Comprehensive Metrics", value=True)
    
    st.markdown("---")
    
    if st.button("🚀 Run Clustering Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

# Initialize session state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Cached Dataset Generation Function
@st.cache_data
def generate_dataset_cached(dataset_type, n_samples, n_features, noise_level, random_state):
    """Generate different types of datasets for clustering with caching support"""
    np.random.seed(random_state)
    
    if dataset_type == "Blobs (Spherical Clusters)":
        estimated_clusters = min(10, max(3, n_samples // 200))
        X, y_true = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=estimated_clusters,
            cluster_std=noise_level * 5 + 0.5,
            random_state=random_state
        )
        dataset_name = f"Blobs Dataset ({estimated_clusters} clusters)"
        
    elif dataset_type == "Moons (Non-linear)":
        X, y_true = make_moons(
            n_samples=n_samples,
            noise=noise_level,
            random_state=random_state
        )
        if n_features > 2:
            extra_dims = np.random.randn(n_samples, n_features - 2) * noise_level
            X = np.hstack([X, extra_dims])
        dataset_name = "Moons Dataset"
        
    elif dataset_type == "Circles (Concentric)":
        X, y_true = make_circles(
            n_samples=n_samples,
            noise=noise_level,
            factor=0.5,
            random_state=random_state
        )
        if n_features > 2:
            extra_dims = np.random.randn(n_samples, n_features - 2) * noise_level
            X = np.hstack([X, extra_dims])
        dataset_name = "Circles Dataset"
        
    elif dataset_type == "Anisotropic (Stretched)":
        X, y_true = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=min(5, n_samples // 200),
            cluster_std=noise_level * 3 + 0.3,
            random_state=random_state
        )
        transformation = np.random.randn(n_features, n_features)
        X = np.dot(X, transformation)
        dataset_name = "Anisotropic Dataset"
        
    elif dataset_type == "Noisy Circles":
        X, y_true = make_circles(
            n_samples=n_samples,
            noise=min(noise_level * 2, 0.3),
            factor=0.3,
            random_state=random_state
        )
        if n_features > 2:
            extra_dims = np.random.randn(n_samples, n_features - 2) * noise_level * 2
            X = np.hstack([X, extra_dims])
        dataset_name = "Noisy Circles Dataset"
    else:
        return None, None, None, None

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    feature_names = [f"Feature {i+1}" for i in range(n_features)]
    return X, y_true, dataset_name, feature_names

# Cached Metrics Calculation
@st.cache_data
def calculate_all_metrics(X, max_clusters, random_state):
    """Calculate multiple metrics for K selection with caching"""
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        
        if len(np.unique(labels)) > 1:
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(0)
        
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
    
    return {
        'k_values': list(range(2, max_clusters + 1)),
        'inertias': inertias,
        'silhouette': silhouette_scores,
        'davies_bouldin': davies_bouldin_scores,
        'calinski_harabasz': calinski_harabasz_scores
    }

def find_optimal_k(metrics):
    """Find optimal K using multiple evaluation metrics"""
    k_values = metrics['k_values']
    inertias = metrics['inertias']
    
    if len(inertias) >= 3:
        second_deriv = np.diff(np.diff(inertias))
        elbow_k = k_values[np.argmax(second_deriv) + 2] if len(second_deriv) > 0 else k_values[0]
    else:
        elbow_k = k_values[0]
    
    silhouette_k = k_values[np.argmax(metrics['silhouette'])]
    db_k = k_values[np.argmin(metrics['davies_bouldin'])]
    ch_k = k_values[np.argmax(metrics['calinski_harabasz'])]
    
    weights = [0.3, 0.3, 0.2, 0.2]
    k_candidates = [elbow_k, silhouette_k, db_k, ch_k]
    recommended_k = int(np.round(np.average(k_candidates, weights=weights)))
    recommended_k = max(min(recommended_k, max(k_values)), min(k_values))
    
    return {
        'elbow': elbow_k,
        'silhouette': silhouette_k,
        'davies_bouldin': db_k,
        'calinski_harabasz': ch_k,
        'recommended': recommended_k
    }

def reduce_dimensionality(X, method='umap', n_components=2):
    """Reduce dimensions for visualization with exception fallback"""
    try:
        if method == 'umap':
            reducer = UMAP(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X)-1))
            X_reduced = reducer.fit_transform(X)
        else:
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
        return X_reduced, reducer
    except Exception as e:
        st.warning(f"Dimensionality reduction ({method}) failed: {e}. Falling back to standard projection.")
        return X[:, :2] if X.shape[1] >= 2 else np.column_stack([X[:, 0], X[:, 0]]), None

def test_cluster_stability(X, n_clusters, n_iterations=10, random_state=42):
    """Test cluster stability using resampling seeds"""
    np.random.seed(random_state)
    all_labels = []
    all_centroids = []
    
    for i in range(n_iterations):
        seed = random_state + i * 100
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X)
        all_labels.append(labels)
        all_centroids.append(kmeans.cluster_centers_)
    
    stability_scores = []
    for i in range(n_iterations):
        for j in range(i+1, n_iterations):
            agreement = np.mean(all_labels[i] == all_labels[j])
            stability_scores.append(agreement)
            
    avg_stability = np.mean(stability_scores) if stability_scores else 0
    
    centroid_variability = []
    for k in range(n_clusters):
        centroids_k = [centroids[k] for centroids in all_centroids]
        if len(centroids_k) > 1:
            variability = np.mean([np.linalg.norm(centroids_k[i] - centroids_k[j]) 
                                 for i in range(len(centroids_k)) for j in range(i+1, len(centroids_k))])
            centroid_variability.append(variability)
            
    avg_centroid_variability = np.mean(centroid_variability) if centroid_variability else 0
    return {'avg_stability': avg_stability, 'avg_centroid_variability': avg_centroid_variability, 'stability_scores': stability_scores}

def discover_subclusters(X, labels, parent_cluster_id, min_samples=50):
    """Discover sub-clusters within a specific parent cluster"""
    cluster_points = X[labels == parent_cluster_id]
    if len(cluster_points) < min_samples * 2:
        return []
    
    max_subclusters = min(5, len(cluster_points) // min_samples)
    best_score = -1
    best_subclusters = None
    best_labels = None
    
    for k in range(2, max_subclusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(cluster_points)
        if len(np.unique(sub_labels)) > 1:
            score = silhouette_score(cluster_points, sub_labels)
            if score > best_score:
                best_score = score
                best_subclusters = k
                best_labels = sub_labels
                
    if best_subclusters and best_score > 0.3:
        subclusters_info = []
        for sub_id in range(best_subclusters):
            sub_points = cluster_points[best_labels == sub_id]
            subclusters_info.append({
                'parent_cluster': parent_cluster_id,
                'subcluster_id': sub_id,
                'size': len(sub_points),
                'density': len(sub_points) / len(cluster_points) * 100,
                'silhouette_score': best_score
            })
        return subclusters_info
    return []

def initialize_llm_client():
    """Safely initialize OpenAI client using Streamlit secrets"""
    if not OPENAI_AVAILABLE:
        return None
    try:
        api_key = st.secrets.get("openai", {}).get("api_key")
        if api_key:
            return openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Failed to load OpenAI API key: {e}")
    return None

def generate_rule_based_labels(X, labels, feature_names, cluster_stats):
    """Fallback rule-based description generator"""
    cluster_labels = []
    for cluster_id in range(len(np.unique(labels))):
        stats = cluster_stats[cluster_id]
        size = stats['size']
        density = stats['density']
        
        size_group = 'Dominant' if density > 30 else ('Significant' if density > 15 else 'Standard')
        name = f"{size_group} Group {cluster_id}"
        explanation = f"Cluster containing {size} records ({density:.1f}% of total data)."
        interpretation = "Pattern identified through distance metrics."
        
        cluster_labels.append({
            'cluster_id': cluster_id,
            'llm_name': name,
            'llm_explanation': explanation,
            'llm_interpretation': interpretation,
            'size': size,
            'density': density,
            'is_rule_based': True
        })
    return cluster_labels

# Main Execution Flow
if st.session_state.run_analysis:
    with st.spinner("🔍 Running data generation and analysis pipeline..."):
        
        # Handle Custom CSV vs Standard Synthetic Datasets
        if dataset_type == "Custom CSV Upload":
            uploaded_file = st.file_uploader("Upload CSV Data File", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) < 2:
                        st.error("Uploaded CSV must contain at least 2 numeric columns.")
                        st.stop()
                    
                    selected_cols = st.multiselect("Select columns for clustering:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
                    if len(selected_cols) < 2:
                        st.warning("Please select at least 2 columns.")
                        st.stop()
                        
                    X = df[selected_cols].values
                    X = np.nan_to_num(X) # Handle missing values
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    y_true = None
                    dataset_name = f"Uploaded CSV: {uploaded_file.name}"
                    feature_names = selected_cols
                except Exception as e:
                    st.error(f"Error parsing uploaded CSV: {e}")
                    st.stop()
            else:
                st.info("Please upload a CSV file via the sidebar or select a synthetic generator.")
                st.stop()
        else:
            progress_bar = st.progress(0)
            X, y_true, dataset_name, feature_names = generate_dataset_cached(
                dataset_type, n_samples, n_features, noise_level, random_state
            )
            progress_bar.progress(100)
        
        if X is not None:
            st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Dataset Overview</h3></div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Samples", X.shape[0])
            col2.metric("Features", X.shape[1])
            col3.metric("Dataset Type", dataset_type.split("(")[0].strip())
            col4.metric("True Clusters", len(np.unique(y_true)) if y_true is not None else "N/A (Custom)")
            
            # K Selection Logic
            if clustering_method == "Manual K":
                optimal_k = n_clusters
            else:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">🎯 Auto-K Selection Analysis</h4></div>', unsafe_allow_html=True)
                metrics = calculate_all_metrics(X, max_clusters, random_state)
                optimal_k_info = find_optimal_k(metrics)
                optimal_k = optimal_k_info['recommended']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Elbow Method", optimal_k_info['elbow'])
                col2.metric("Silhouette", optimal_k_info['silhouette'])
                col3.metric("Davies-Bouldin", optimal_k_info['davies_bouldin'])
                col4.metric("Calinski-Harabasz", optimal_k_info['calinski_harabasz'])
                col5.metric("🚀 Recommended K", optimal_k)
                
                # Plot selection curves
                fig = make_subplots(rows=2, cols=2, subplot_titles=('Elbow Method', 'Silhouette Scores', 'Davies-Bouldin', 'Calinski-Harabasz'))
                fig.add_trace(go.Scatter(x=metrics['k_values'], y=metrics['inertias'], mode='lines+markers', name='Inertia'), row=1, col=1)
                fig.add_trace(go.Scatter(x=metrics['k_values'], y=metrics['silhouette'], mode='lines+markers', name='Silhouette'), row=1, col=2)
                fig.add_trace(go.Scatter(x=metrics['k_values'], y=metrics['davies_bouldin'], mode='lines+markers', name='DB'), row=2, col=1)
                fig.add_trace(go.Scatter(x=metrics['k_values'], y=metrics['calinski_harabasz'], mode='lines+markers', name='CH'), row=2, col=2)
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Fit final K-Means model
            kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=CONFIG["default_n_init"])
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            
            st.markdown(f'<div class="section-header"><h4 style="margin: 0;">🎯 Clustering Model Performance (K={optimal_k})</h4></div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Inertia", f"{kmeans.inertia_:,.0f}")
            col2.metric("Silhouette Score", f"{silhouette_score(X, labels):.3f}")
            col3.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X, labels):.3f}")
            col4.metric("Calinski-Harabasz Index", f"{calinski_harabasz_score(X, labels):,.0f}")
            
            # Tabs for deep-dive diagnostics
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📐 Multi-dimensional View", 
                "⚖️ Stability Analysis", 
                "🔍 Sub-cluster Discovery",
                "📋 Cluster Profiles",
                "📥 Export Results"
            ])
            
            with tab1:
                if include_dimensionality:
                    reduction_method = st.selectbox("Reduction Algorithm:", ["UMAP", "t-SNE", "PCA"])
                    X_reduced, _ = reduce_dimensionality(X, reduction_method.lower())
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_2d = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=labels.astype(str), title=f"{reduction_method} 2D Space")
                        st.plotly_chart(fig_2d, use_container_width=True)
                    with c2:
                        if X_reduced.shape[1] >= 3:
                            fig_3d = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2], color=labels.astype(str), title=f"{reduction_method} 3D Space")
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.info("Insufficient dimensionality for 3D projection.")
                            
            with tab2:
                if include_stability:
                    stability_results = test_cluster_stability(X, optimal_k, n_iterations=10, random_state=random_state)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Average Stability Agreement", f"{stability_results['avg_stability']:.3f}")
                        st.metric("Centroid Variability", f"{stability_results['avg_centroid_variability']:.3f}")
                    with c2:
                        fig_s, ax = plt.subplots(figsize=(6, 3))
                        ax.hist(stability_results['stability_scores'], bins=10, color='skyblue', edgecolor='black')
                        ax.set_title("Stability Distribution")
                        st.pyplot(fig_s)
                        
            with tab3:
                if include_subclusters:
                    sub_list = []
                    for cid in range(optimal_k):
                        sub_list.extend(discover_subclusters(X, labels, cid, min_samples=30))
                    if sub_list:
                        st.dataframe(pd.DataFrame(sub_list), use_container_width=True)
                    else:
                        st.info("No significant sub-clusters detected.")
                        
            with tab4:
                cluster_stats = []
                for cid in range(optimal_k):
                    cpts = X[labels == cid]
                    cluster_stats.append({
                        'cluster_id': cid,
                        'size': len(cpts),
                        'density': len(cpts) / len(X) * 100,
                        'centroid': np.mean(cpts, axis=0) if len(cpts) > 0 else np.zeros(X.shape[1])
                    })
                
                # Initialize LLM client dynamically (Bug fixed here)
                llm_client = initialize_llm_client()
                if include_llm_labeling and llm_client:
                    st.success("🤖 AI LLM Integration active.")
                    # Fallback / placeholder output display for cluster groupings
                else:
                    st.info("Displaying rule-based cluster diagnostics profiles.")
                    rule_labels = generate_rule_based_labels(X, labels, feature_names, cluster_stats)
                    for rl in rule_labels:
                        st.write(f"**{rl['llm_name']}**: {rl['llm_explanation']} ({rl['llm_interpretation']})")
                        
            with tab5:
                st.markdown("##### 📥 Export Options")
                df_exp = pd.DataFrame(X, columns=feature_names)
                df_exp['Cluster'] = labels
                st.download_button("Download Clustered CSV", df_exp.to_csv(index=False).encode('utf-8'), file_name="clustered_output.csv", mime="text/csv")
                
        else:
            st.error("Failed to generate or load dataset. Please check parameters.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem; background: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
        <h3>🚀 Ready to Cluster</h3>
        <p>Configure parameters on the sidebar and click <b>Run Clustering Analysis</b> to initialize the pipeline.</p>
    </div>
    """, unsafe_allow_html=True)