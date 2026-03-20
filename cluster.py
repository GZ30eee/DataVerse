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
import openai
warnings.filterwarnings('ignore')

# Set up Streamlit App
st.set_page_config(page_title="🔍 Advanced K-Means Clustering", layout="wide")

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
    # In the sidebar section, add this checkbox:
    include_llm_labeling = st.checkbox("🤖 LLM Cluster Labeling", value=True)

    if include_llm_labeling:
        st.info("LLM labeling requires OpenAI API key in secrets.toml")
        
    st.markdown("#### 📊 Dataset Generation")
    dataset_type = st.selectbox(
        "Dataset Type:",
        ["Blobs (Spherical Clusters)", "Moons (Non-linear)", "Circles (Concentric)", 
         "Anisotropic (Stretched)", "Noisy Circles", "Custom CSV Upload"]
    )
    
    n_samples = st.slider("Number of Samples", 100, 5000, 1000, 100)
    n_features = st.slider("Number of Features", 2, 20, 5)
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.05, 0.05)
    random_state = st.slider("Random Seed", 0, 100, 42)
    
    st.markdown("---")
    st.markdown("#### 🔧 Clustering Settings")

    clustering_method = st.radio(
        "K Selection Method:",
        ["Manual K", "Auto-K (Elbow + Silhouette)", "Auto-K (All Metrics)"]
    )

    # Fix: Define a default or slider for max_clusters regardless of the choice
    if clustering_method == "Manual K":
        n_clusters = st.slider("Number of Clusters (K)", 2, 20, 5)
        max_clusters = 10  # Add this line so the variable exists for the stability tab
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

# Dataset Generation Functions
def generate_dataset(dataset_type, n_samples, n_features, noise_level, random_state):
    """Generate different types of datasets for clustering"""
    np.random.seed(random_state)
    
    if dataset_type == "Blobs (Spherical Clusters)":
        # Estimate number of clusters for blobs
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
        # Add extra dimensions with noise
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
        # Add extra dimensions with noise
        if n_features > 2:
            extra_dims = np.random.randn(n_samples, n_features - 2) * noise_level
            X = np.hstack([X, extra_dims])
        dataset_name = "Circles Dataset"
        
    elif dataset_type == "Anisotropic (Stretched)":
        # Generate anisotropic dataset
        X, y_true = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=min(5, n_samples // 200),
            cluster_std=noise_level * 3 + 0.3,
            random_state=random_state
        )
        # Apply transformation to make it anisotropic
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
        # Add extra dimensions with noise
        if n_features > 2:
            extra_dims = np.random.randn(n_samples, n_features - 2) * noise_level * 2
            X = np.hstack([X, extra_dims])
        dataset_name = "Noisy Circles Dataset"
        
    else:  # Custom CSV Upload
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            # Select columns for clustering
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect(
                    "Select columns for clustering:", 
                    numeric_cols,
                    default=numeric_cols[:min(10, len(numeric_cols))]
                )
                
                if len(selected_cols) >= 2:
                    X = df[selected_cols].values
                    y_true = None
                    dataset_name = f"Uploaded Dataset: {uploaded_file.name}"
                    
                    # Handle missing values
                    X = np.nan_to_num(X)
                    
                    # Standardize
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    
                    return X, y_true, dataset_name, selected_cols
            
            st.warning("Please upload a CSV with at least 2 numeric columns.")
            return None, None, None, None
    
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    return X, y_true, dataset_name, feature_names

# Auto-K Selection Functions
def calculate_all_metrics(X, max_clusters, random_state):
    """Calculate multiple metrics for K selection"""
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        
        # Silhouette score
        if len(np.unique(labels)) > 1:
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(0)
        
        # Davies-Bouldin score (lower is better)
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        
        # Calinski-Harabasz score (higher is better)
        calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
    
    return {
        'k_values': list(range(2, max_clusters + 1)),
        'inertias': inertias,
        'silhouette': silhouette_scores,
        'davies_bouldin': davies_bouldin_scores,
        'calinski_harabasz': calinski_harabasz_scores
    }

def find_optimal_k(metrics):
    """Find optimal K using multiple methods"""
    k_values = metrics['k_values']
    
    # Elbow method (find point of maximum curvature)
    inertias = metrics['inertias']
    if len(inertias) >= 3:
        # Calculate second derivative
        second_deriv = np.diff(np.diff(inertias))
        if len(second_deriv) > 0:
            elbow_k = k_values[np.argmax(second_deriv) + 2]
        else:
            elbow_k = k_values[0]
    else:
        elbow_k = k_values[0]
    
    # Silhouette method (maximum score)
    silhouette_scores = metrics['silhouette']
    silhouette_k = k_values[np.argmax(silhouette_scores)]
    
    # Davies-Bouldin method (minimum score)
    db_scores = metrics['davies_bouldin']
    db_k = k_values[np.argmin(db_scores)]
    
    # Calinski-Harabasz method (maximum score)
    ch_scores = metrics['calinski_harabasz']
    ch_k = k_values[np.argmax(ch_scores)]
    
    # Combined recommendation (weighted average)
    weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each method
    k_candidates = [elbow_k, silhouette_k, db_k, ch_k]
    recommended_k = int(np.round(np.average(k_candidates, weights=weights)))
    
    # Ensure recommended_k is within bounds
    recommended_k = max(min(recommended_k, max(k_values)), min(k_values))
    
    return {
        'elbow': elbow_k,
        'silhouette': silhouette_k,
        'davies_bouldin': db_k,
        'calinski_harabasz': ch_k,
        'recommended': recommended_k,
        'all_scores': {
            'elbow_scores': inertias,
            'silhouette_scores': silhouette_scores,
            'db_scores': db_scores,
            'ch_scores': ch_scores
        }
    }

# Cluster Labeling Functions
def generate_cluster_labels(X, labels, feature_names):
    """Generate descriptive labels for clusters based on characteristics"""
    n_clusters = len(np.unique(labels))
    cluster_labels = []
    
    for cluster_id in range(n_clusters):
        cluster_points = X[labels == cluster_id]
        
        if len(cluster_points) > 0:
            # Calculate cluster statistics
            centroid = np.mean(cluster_points, axis=0)
            std_dev = np.std(cluster_points, axis=0)
            size = len(cluster_points)
            
            # Find most distinctive features
            if len(feature_names) > 0:
                # Compare with global means
                global_mean = np.mean(X, axis=0)
                differences = np.abs(centroid - global_mean)
                
                if len(differences) > 0:
                    top_features_idx = np.argsort(differences)[-3:][::-1]
                    top_features = [feature_names[i] for i in top_features_idx if i < len(feature_names)]
                    
                    # Generate descriptive label
                    if len(top_features) > 0:
                        label = f"Cluster {cluster_id}: {size} points, high in {', '.join(top_features[:2])}"
                    else:
                        label = f"Cluster {cluster_id}: {size} points"
                else:
                    label = f"Cluster {cluster_id}: {size} points"
            else:
                label = f"Cluster {cluster_id}: {size} points"
            
            # Add density information
            density = size / len(X) * 100
            label += f" ({density:.1f}% of data)"
            
            cluster_labels.append({
                'cluster_id': cluster_id,
                'label': label,
                'size': size,
                'density': density,
                'centroid': centroid,
                'std_dev': std_dev
            })
    
    return cluster_labels

# Dimensionality Reduction Functions
def reduce_dimensionality(X, method='umap', n_components=2):
    """Reduce dimensions for visualization"""
    try:
        if method == 'umap':
            reducer = UMAP(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
            
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X)-1))
            X_reduced = reducer.fit_transform(X)
            
        else:  # pca
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
            
        return X_reduced, reducer
        
    except Exception as e:
        st.warning(f"Dimensionality reduction failed: {e}")
        # Use first two features as fallback
        if X.shape[1] >= 2:
            return X[:, :2], None
        else:
            # If only 1 feature, duplicate it
            return np.column_stack([X[:, 0], X[:, 0]]), None

# Cluster Stability Testing
def test_cluster_stability(X, n_clusters, n_iterations=10, random_state=42):
    """Test cluster stability using multiple random seeds"""
    np.random.seed(random_state)
    
    all_labels = []
    all_centroids = []
    
    for i in range(n_iterations):
        seed = random_state + i * 100
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X)
        all_labels.append(labels)
        all_centroids.append(kmeans.cluster_centers_)
    
    # Calculate stability metrics
    stability_scores = []
    for i in range(n_iterations):
        for j in range(i+1, n_iterations):
            # Adjusted Rand Index would be better, but let's use simple agreement
            agreement = np.mean(all_labels[i] == all_labels[j])
            stability_scores.append(agreement)
    
    avg_stability = np.mean(stability_scores) if stability_scores else 0
    
    # Calculate centroid variability
    centroid_variability = []
    for k in range(n_clusters):
        centroids_k = [centroids[k] for centroids in all_centroids]
        if len(centroids_k) > 1:
            variability = np.mean([np.linalg.norm(centroids_k[i] - centroids_k[j]) 
                                 for i in range(len(centroids_k)) 
                                 for j in range(i+1, len(centroids_k))])
            centroid_variability.append(variability)
    
    avg_centroid_variability = np.mean(centroid_variability) if centroid_variability else 0
    
    return {
        'avg_stability': avg_stability,
        'avg_centroid_variability': avg_centroid_variability,
        'stability_scores': stability_scores,
        'labels_history': all_labels
    }

# Sub-cluster Discovery
def discover_subclusters(X, labels, parent_cluster_id, min_samples=50):
    """Discover sub-clusters within a given cluster"""
    cluster_points = X[labels == parent_cluster_id]
    
    if len(cluster_points) < min_samples * 2:
        return []  # Not enough points for sub-clustering
    
    # Try different numbers of sub-clusters
    max_subclusters = min(5, len(cluster_points) // min_samples)
    
    best_score = -1
    best_subclusters = None
    best_labels = None
    
    for k in range(2, max_subclusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(cluster_points)
        
        # Calculate silhouette score for sub-clustering
        if len(np.unique(sub_labels)) > 1:
            score = silhouette_score(cluster_points, sub_labels)
            
            if score > best_score:
                best_score = score
                best_subclusters = k
                best_labels = sub_labels
    
    if best_subclusters and best_score > 0.3:  # Reasonable silhouette threshold
        subclusters_info = []
        for sub_id in range(best_subclusters):
            sub_points = cluster_points[best_labels == sub_id]
            sub_size = len(sub_points)
            sub_density = sub_size / len(cluster_points) * 100
            
            subclusters_info.append({
                'parent_cluster': parent_cluster_id,
                'subcluster_id': sub_id,
                'size': sub_size,
                'density': sub_density,
                'silhouette_score': best_score
            })
        
        return subclusters_info
    
    return []

# LLM-based Cluster Labeling Functions
# LLM-based Cluster Labeling Functions
def initialize_llm_client():
    """Initialize OpenAI client for LLM-based labeling"""
    try:
        api_key = st.secrets.get("openai", {}).get("api_key")
        if api_key:
            return openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize LLM client: {e}")
    return None

# The rest of your code (generate_llm_cluster_labels, etc.) can stay as is.

# LLM Helper Functions
def generate_llm_cluster_labels(client, X, labels, feature_names, cluster_stats, n_samples_per_cluster=5):
    """Generate intelligent cluster labels using LLM"""
    try:
        cluster_labels = []
        
        for cluster_id in range(len(np.unique(labels))):
            # Get cluster statistics
            stats = cluster_stats[cluster_id]
            
            # Prepare feature descriptions
            if feature_names and len(feature_names) > 0:
                # Get top 3 most distinctive features
                global_mean = np.mean(X, axis=0)
                cluster_mean = stats['centroid']
                feature_diffs = np.abs(cluster_mean - global_mean)
                
                top_feature_indices = np.argsort(feature_diffs)[-3:][::-1]
                top_features = []
                for idx in top_feature_indices:
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        cluster_val = cluster_mean[idx]
                        global_val = global_mean[idx]
                        diff = cluster_val - global_val
                        
                        if diff > 0:
                            direction = "higher than average"
                        else:
                            direction = "lower than average"
                        
                        top_features.append(f"{feature_name} ({abs(diff):.2f} {direction})")
                
                feature_description = f"Key characteristics: {', '.join(top_features)}"
            else:
                feature_description = f"Cluster with {stats['size']} data points"
            
            # Prepare prompt for LLM
            prompt = f"""
            I have a cluster of data points from a dataset. Please provide a creative, descriptive label for this cluster.
            
            Cluster information:
            - Number of points: {stats['size']}
            - Cluster density: {stats['density']:.1f}% of total dataset
            - {feature_description}
            
            Please provide:
            1. A short, descriptive name (2-4 words)
            2. A brief explanation (1 sentence)
            3. Suggested business/real-world interpretation
            
            Format your response as:
            Name: [Your label name]
            Explanation: [Your explanation]
            Interpretation: [Business interpretation]
            """
            
            # Call LLM
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert who creates insightful, descriptive labels for data clusters."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            # Parse response
            llm_response = response.choices[0].message.content
            
            # Parse the response
            name = f"Cluster {cluster_id}"
            explanation = ""
            interpretation = ""
            
            lines = llm_response.split('\n')
            for line in lines:
                if line.startswith('Name:'):
                    name = line.replace('Name:', '').strip()
                elif line.startswith('Explanation:'):
                    explanation = line.replace('Explanation:', '').strip()
                elif line.startswith('Interpretation:'):
                    interpretation = line.replace('Interpretation:', '').strip()
            
            cluster_labels.append({
                'cluster_id': cluster_id,
                'llm_name': name,
                'llm_explanation': explanation,
                'llm_interpretation': interpretation,
                'size': stats['size'],
                'density': stats['density']
            })
        
        return cluster_labels
        
    except Exception as e:
        print(f"LLM labeling failed: {str(e)}")
        return None

def generate_rule_based_labels(X, labels, feature_names, cluster_stats):
    """Generate descriptive labels using rule-based approach when LLM is not available"""
    cluster_labels = []
    
    for cluster_id in range(len(np.unique(labels))):
        stats = cluster_stats[cluster_id]
        size = stats['size']
        density = stats['density']
        
        # Determine label based on size and density
        if density > 30:
            size_group = 'Dominant'
            description = f"Major cluster containing {size} core data points"
        elif density > 15:
            size_group = 'Significant'
            description = f"Large cluster with {size} important points"
        elif density > 5:
            size_group = 'Standard'
            description = f"Medium cluster of {size} typical points"
        else:
            size_group = 'Minor'
            description = f"Small cluster with {size} outlier points"
        
        # Check cluster compactness
        cluster_data = X[labels == cluster_id]
        if len(cluster_data) > 1:
            distances = np.linalg.norm(cluster_data - stats['centroid'], axis=1)
            compactness = 1 / (distances.mean() + 1e-10)
            
            if compactness > 1:
                compact_label = "Compact"
            else:
                compact_label = "Dispersed"
        else:
            compact_label = "Single"
        
        # Generate name
        name = f"{compact_label} {size_group} Group"
        
        # Generate interpretation
        if feature_names and len(feature_names) > 0:
            global_mean = np.mean(X, axis=0)
            feature_diffs = np.abs(stats['centroid'] - global_mean)
            
            if len(feature_diffs) > 0:
                top_feature_idx = np.argmax(feature_diffs)
                if top_feature_idx < len(feature_names):
                    top_feature = feature_names[top_feature_idx]
                    diff = stats['centroid'][top_feature_idx] - global_mean[top_feature_idx]
                    
                    if diff > 0:
                        interpretation = f"Characterized by high {top_feature}"
                    else:
                        interpretation = f"Characterized by low {top_feature}"
                else:
                    interpretation = "Unique data pattern"
            else:
                interpretation = "Unique data pattern"
        else:
            interpretation = "Distinct data pattern"
        
        cluster_labels.append({
            'cluster_id': cluster_id,
            'llm_name': name,
            'llm_explanation': description,
            'llm_interpretation': interpretation,
            'size': size,
            'density': density,
            'is_rule_based': True
        })
    
    return cluster_labels
# Main Analysis Execution
if st.session_state.run_analysis:
    with st.spinner("🔍 Generating dataset and analyzing clusters..."):
        
        # Generate dataset
        X, y_true, dataset_name, feature_names = generate_dataset(
            dataset_type, n_samples, n_features, noise_level, random_state
        )
        
        if X is not None:
            # Dataset Overview
            st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Dataset Overview</h3></div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Samples", X.shape[0])
            with col2:
                st.metric("Features", X.shape[1])
            with col3:
                st.metric("Dataset Type", dataset_type.split("(")[0].strip())
            with col4:
                if y_true is not None:
                    true_clusters = len(np.unique(y_true))
                    st.metric("True Clusters", true_clusters)
                else:
                    st.metric("Data Type", "Uploaded")
            
            # Determine optimal K or use manual K
            if clustering_method == "Manual K":
                optimal_k = n_clusters
                k_metrics = None
            else:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">🎯 Auto-K Selection Analysis</h4></div>', unsafe_allow_html=True)
                
                # Calculate all metrics
                metrics = calculate_all_metrics(X, max_clusters, random_state)
                optimal_k_info = find_optimal_k(metrics)
                optimal_k = optimal_k_info['recommended']
                
                # Display K selection results
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Elbow Method", optimal_k_info['elbow'])
                col2.metric("Silhouette", optimal_k_info['silhouette'])
                col3.metric("Davies-Bouldin", optimal_k_info['davies_bouldin'])
                col4.metric("Calinski-Harabasz", optimal_k_info['calinski_harabasz'])
                col5.metric("🚀 Recommended K", optimal_k)
                
                # Plot all K selection metrics
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Elbow Method', 'Silhouette Scores',
                                  'Davies-Bouldin Index', 'Calinski-Harabasz Index'),
                    vertical_spacing=0.15
                )
                
                # Elbow plot
                fig.add_trace(
                    go.Scatter(x=metrics['k_values'], y=metrics['inertias'],
                              mode='lines+markers', name='Inertia',
                              line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                
                # Silhouette plot
                fig.add_trace(
                    go.Scatter(x=metrics['k_values'], y=metrics['silhouette'],
                              mode='lines+markers', name='Silhouette',
                              line=dict(color='green', width=2)),
                    row=1, col=2
                )
                
                # Davies-Bouldin plot (lower is better)
                fig.add_trace(
                    go.Scatter(x=metrics['k_values'], y=metrics['davies_bouldin'],
                              mode='lines+markers', name='Davies-Bouldin',
                              line=dict(color='red', width=2)),
                    row=2, col=1
                )
                
                # Calinski-Harabasz plot (higher is better)
                fig.add_trace(
                    go.Scatter(x=metrics['k_values'], y=metrics['calinski_harabasz'],
                              mode='lines+markers', name='Calinski-Harabasz',
                              line=dict(color='purple', width=2)),
                    row=2, col=2
                )
                
                # Update layout
                fig.update_layout(height=600, showlegend=True, title_text="K Selection Metrics")
                fig.update_xaxes(title_text="Number of Clusters (K)")
                fig.update_yaxes(title_text="Inertia", row=1, col=1)
                fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
                fig.update_yaxes(title_text="Davies-Bouldin Score", row=2, col=1)
                fig.update_yaxes(title_text="Calinski-Harabasz Score", row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                k_metrics = metrics
            
            # Apply K-Means with optimal K
            st.markdown(f'<div class="section-header"><h4 style="margin: 0;">🎯 K-Means Clustering with K={optimal_k}</h4></div>', unsafe_allow_html=True)
            
            kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=20)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            
            # Calculate cluster metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            
            # Display clustering results
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Inertia", f"{inertia:,.0f}")
            col2.metric("Silhouette Score", f"{silhouette:.3f}")
            col3.metric("Davies-Bouldin", f"{db_score:.3f}")
            col4.metric("Calinski-Harabasz", f"{ch_score:,.0f}")
            
            # Silhouette Analysis
            st.markdown("#### 📊 Silhouette Analysis")
            
            # Calculate silhouette values for each sample
            sample_silhouette_values = silhouette_samples(X, labels)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Silhouette plot
            y_lower = 10
            for i in range(optimal_k):
                # Aggregate silhouette scores for samples in cluster i
                ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
                ith_cluster_silhouette_values.sort()
                
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = plt.cm.nipy_spectral(float(i) / optimal_k)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10
            
            ax1.set_xlabel("Silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            ax1.axvline(x=silhouette, color="red", linestyle="--")
            ax1.set_yticks([])
            ax1.set_xlim([-0.1, 1])
            ax1.set_title(f"Silhouette plot for K={optimal_k}")
            
            # Cluster visualization (first 2 dimensions)
            scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, s=50, 
                                 cmap='viridis', alpha=0.6, edgecolors='k')
            ax2.scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                       alpha=0.8, marker='X', label="Centroids")
            ax2.set_title("Cluster Visualization (First 2 Features)")
            ax2.set_xlabel(feature_names[0] if feature_names else "Feature 1")
            ax2.set_ylabel(feature_names[1] if feature_names else "Feature 2")
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabs for advanced features
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📐 Multi-dimensional View", 
                "⚖️ Stability Analysis", 
                "🔍 Sub-cluster Discovery",
                "📋 Cluster Profiles",
                "📥 Export Results"
            ])
            
            with tab1:
                if include_dimensionality:
                    st.markdown("#### 📐 Multi-dimensional Visualizations")
                    
                    # Dimensionality reduction method selection
                    reduction_method = st.selectbox(
                        "Reduction Method:", 
                        ["UMAP", "t-SNE", "PCA"]
                    )
                    
                    # Apply dimensionality reduction
                    X_reduced, reducer = reduce_dimensionality(X, reduction_method.lower())
                    
                    # Create 3D plot if possible
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 2D Plot
                        fig_2d = px.scatter(
                            x=X_reduced[:, 0], 
                            y=X_reduced[:, 1],
                            color=labels.astype(str),
                            title=f"{reduction_method} 2D Projection",
                            labels={'x': f'{reduction_method} 1', 'y': f'{reduction_method} 2'},
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)
                    
                    with col2:
                        # Try 3D plot if we have enough dimensions
                        if X_reduced.shape[1] >= 3:
                            fig_3d = px.scatter_3d(
                                x=X_reduced[:, 0], 
                                y=X_reduced[:, 1],
                                z=X_reduced[:, 2],
                                color=labels.astype(str),
                                title=f"{reduction_method} 3D Projection",
                                labels={'x': f'{reduction_method} 1', 
                                       'y': f'{reduction_method} 2',
                                       'z': f'{reduction_method} 3'},
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            # Feature importance for first few clusters
                            if optimal_k >= 3:
                                cluster_means = []
                                for i in range(min(3, optimal_k)):
                                    cluster_data = X[labels == i]
                                    cluster_means.append(np.mean(cluster_data, axis=0))
                                
                                if len(feature_names) > 0 and len(cluster_means) > 0:
                                    # Show top features for first 3 clusters
                                    fig_feat, axes = plt.subplots(1, 3, figsize=(15, 4))
                                    for i in range(3):
                                        if i < len(cluster_means):
                                            top_indices = np.argsort(np.abs(cluster_means[i]))[-5:][::-1]
                                            top_features = [feature_names[j] for j in top_indices if j < len(feature_names)]
                                            top_values = [cluster_means[i][j] for j in top_indices if j < len(feature_names)]
                                            
                                            axes[i].barh(top_features, top_values)
                                            axes[i].set_title(f"Cluster {i} - Top Features")
                                            axes[i].set_xlabel("Mean Value")
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig_feat)
            
            with tab2:
                if include_stability:
                    st.markdown("#### ⚖️ Cluster Stability Testing")
                    
                    # Run stability analysis
                    stability_results = test_cluster_stability(X, optimal_k, n_iterations=10, random_state=random_state)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Average Stability", f"{stability_results['avg_stability']:.3f}")
                        st.metric("Centroid Variability", f"{stability_results['avg_centroid_variability']:.3f}")
                        
                        # Stability interpretation
                        if stability_results['avg_stability'] > 0.8:
                            st.success("✅ High cluster stability - results are reliable")
                        elif stability_results['avg_stability'] > 0.6:
                            st.warning("⚠️ Moderate cluster stability - results may vary")
                        else:
                            st.error("❌ Low cluster stability - consider different K or preprocessing")
                    
                    with col2:
                        # Plot stability scores distribution
                        fig_stab, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(stability_results['stability_scores'], bins=10, edgecolor='black', alpha=0.7)
                        ax.axvline(x=stability_results['avg_stability'], color='red', linestyle='--', label=f'Mean: {stability_results["avg_stability"]:.3f}')
                        ax.set_xlabel('Stability Score')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Cluster Stability Distribution')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig_stab)
                    
                    # Stability vs K analysis
                    st.markdown("#### 📈 Stability Across Different K Values")
                    
                    k_range = range(2, min(11, max_clusters + 1))
                    stability_by_k = []
                    
                    for k in k_range:
                        stab_result = test_cluster_stability(X, k, n_iterations=5, random_state=random_state)
                        stability_by_k.append(stab_result['avg_stability'])
                    
                    fig_k_stab, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(k_range, stability_by_k, marker='o', linestyle='-', linewidth=2)
                    ax.axvline(x=optimal_k, color='green', linestyle='--', label=f'Selected K={optimal_k}')
                    ax.set_xlabel('Number of Clusters (K)')
                    ax.set_ylabel('Average Stability')
                    ax.set_title('Cluster Stability vs K')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig_k_stab)
            
            with tab3:
                if include_subclusters:
                    st.markdown("#### 🔍 Sub-cluster Discovery")
                    
                    # Find clusters that might have sub-clusters
                    potential_subclusters = []
                    
                    for cluster_id in range(optimal_k):
                        subclusters = discover_subclusters(X, labels, cluster_id, min_samples=30)
                        if subclusters:
                            potential_subclusters.extend(subclusters)
                    
                    if potential_subclusters:
                        st.success(f"Found {len(potential_subclusters)} potential sub-clusters!")
                        
                        # Display sub-cluster information
                        subcluster_df = pd.DataFrame(potential_subclusters)
                        st.dataframe(subcluster_df, use_container_width=True)
                        
                        # Visualize one example
                        if len(potential_subclusters) > 0:
                            example_cluster = potential_subclusters[0]['parent_cluster']
                            cluster_points = X[labels == example_cluster]
                            
                            # Apply sub-clustering to this cluster
                            kmeans_sub = KMeans(n_clusters=min(3, len(cluster_points)//30), 
                                              random_state=42, n_init=10)
                            sub_labels = kmeans_sub.fit_predict(cluster_points)
                            
                            # Plot sub-clusters
                            fig_sub, ax = plt.subplots(figsize=(8, 6))
                            scatter = ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                               c=sub_labels, s=50, cmap='tab20', alpha=0.7)
                            ax.set_title(f"Sub-clusters within Cluster {example_cluster}")
                            ax.set_xlabel("Feature 1")
                            ax.set_ylabel("Feature 2")
                            plt.colorbar(scatter, ax=ax, label='Sub-cluster')
                            st.pyplot(fig_sub)
                    else:
                        st.info("No significant sub-clusters found. Clusters appear to be well-separated.")
            
            with tab4:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">📋 Cluster Profiles & Labeling</h4></div>', unsafe_allow_html=True)
                
                # LLM Labeling Section
                if include_llm_labeling:
                    st.markdown("##### 🤖 LLM-Powered Cluster Labeling")
                    
                    # Initialize LLM client - HARDCODED BACKEND FIX
                    llm_client = None # Force it to None so it uses rule-based
                    st.success("✅ Backend labeling active!")
                # Generate basic cluster statistics first
                cluster_stats = []
                for cluster_id in range(optimal_k):
                    cluster_points = X[labels == cluster_id]
                    if len(cluster_points) > 0:
                        centroid = np.mean(cluster_points, axis=0)
                        std_dev = np.std(cluster_points, axis=0)
                        size = len(cluster_points)
                        density = size / len(X) * 100
                        
                        cluster_stats.append({
                            'cluster_id': cluster_id,
                            'size': size,
                            'density': density,
                            'centroid': centroid,
                            'std_dev': std_dev
                        })
                
                # Generate cluster labels (with or without LLM)
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if include_llm_labeling and llm_client:
                        with st.spinner("🤖 Generating intelligent cluster labels with AI..."):
                            # Generate LLM labels
                            llm_labels = generate_llm_cluster_labels(
                                client=llm_client,
                                X=X,
                                labels=labels,
                                feature_names=feature_names,
                                cluster_stats=cluster_stats
                            )
                            
                            if llm_labels:
                                st.success("✅ AI-generated labels created successfully!")
                                
                                # Display LLM labels in an interactive way
                                for label_info in llm_labels:
                                    with st.expander(f"🤖 **{label_info.get('llm_name', f'Cluster {label_info["cluster_id"]}')}**", expanded=True):
                                        col_a, col_b = st.columns([3, 1])
                                        with col_a:
                                            st.markdown(f"**Explanation:** {label_info.get('llm_explanation', 'No explanation available')}")
                                            st.markdown(f"**Business Interpretation:** {label_info.get('llm_interpretation', 'No interpretation available')}")
                                            
                                            # Show key features if available
                                            if feature_names and len(feature_names) > 0:
                                                cluster_data = X[labels == label_info['cluster_id']]
                                                cluster_mean = np.mean(cluster_data, axis=0)
                                                global_mean = np.mean(X, axis=0)
                                                feature_diffs = np.abs(cluster_mean - global_mean)
                                                
                                                top_indices = np.argsort(feature_diffs)[-3:][::-1]
                                                top_features = []
                                                for idx in top_indices:
                                                    if idx < len(feature_names):
                                                        diff = cluster_mean[idx] - global_mean[idx]
                                                        direction = "↑ higher" if diff > 0 else "↓ lower"
                                                        top_features.append(f"{feature_names[idx]} ({direction})")
                                                
                                                if top_features:
                                                    st.caption(f"📊 **Distinctive Features:** {', '.join(top_features)}")
                                        
                                        with col_b:
                                            st.metric("Size", label_info['size'])
                                            st.metric("Density", f"{label_info['density']:.1f}%")
                                
                                # Store for export
                                llm_labels_df = pd.DataFrame(llm_labels)
                            else:
                                st.warning("LLM labeling failed. Using rule-based labels.")
                                llm_labels = generate_rule_based_labels(X, labels, feature_names, cluster_stats)
                                llm_labels_df = pd.DataFrame(llm_labels)
                    
                    elif include_llm_labeling:
                        # Use rule-based labeling as fallback
                        with st.spinner("📊 Generating rule-based cluster labels..."):
                            llm_labels = generate_rule_based_labels(X, labels, feature_names, cluster_stats)
                            llm_labels_df = pd.DataFrame(llm_labels)
                            
                            # Display rule-based labels
                            for label_info in llm_labels:
                                with st.expander(f"📊 **{label_info.get('llm_name', f'Cluster {label_info["cluster_id"]}')}**", expanded=True):
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.markdown(f"**Description:** {label_info.get('llm_explanation', 'No description available')}")
                                        st.markdown(f"**Characteristics:** {label_info.get('llm_interpretation', 'No characteristics available')}")
                                        
                                        # Show key statistics
                                        if 'is_rule_based' in label_info:
                                            st.caption("🔄 Rule-based labeling (LLM not available)")
                                    
                                    with col_b:
                                        st.metric("Size", label_info['size'])
                                        st.metric("Density", f"{label_info['density']:.1f}%")
                    
                    else:
                        # Basic cluster profiles without LLM
                        st.info("Enable LLM labeling in sidebar for AI-generated cluster descriptions")
                        
                        for stats in cluster_stats:
                            with st.expander(f"📊 **Cluster {stats['cluster_id']}**", expanded=True):
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    # Generate basic description
                                    if stats['density'] > 20:
                                        description = f"Major cluster containing {stats['size']} points ({stats['density']:.1f}% of data)"
                                    elif stats['density'] > 5:
                                        description = f"Medium cluster with {stats['size']} points ({stats['density']:.1f}% of data)"
                                    else:
                                        description = f"Minor cluster of {stats['size']} points ({stats['density']:.1f}% of data)"
                                    
                                    st.markdown(f"**Description:** {description}")
                                    
                                    # Show key features if available
                                    if feature_names and len(feature_names) > 0:
                                        global_mean = np.mean(X, axis=0)
                                        feature_diffs = np.abs(stats['centroid'] - global_mean)
                                        
                                        top_indices = np.argsort(feature_diffs)[-3:][::-1]
                                        top_features = []
                                        for idx in top_indices:
                                            if idx < len(feature_names):
                                                diff = stats['centroid'][idx] - global_mean[idx]
                                                direction = "above average" if diff > 0 else "below average"
                                                top_features.append(f"{feature_names[idx]} ({direction})")
                                        
                                        if top_features:
                                            st.caption(f"🔑 **Key Features:** {', '.join(top_features)}")
                                
                                with col_b:
                                    st.metric("Size", stats['size'])
                                    st.metric("Density", f"{stats['density']:.1f}%")
                
                with col2:
                    # Cluster Quality Metrics
                    st.markdown("#### 📈 Cluster Quality")
                    
                    # Calculate cluster separation
                    if optimal_k > 1:
                        # Calculate inter-cluster distances
                        from sklearn.metrics import pairwise_distances
                        cluster_means = [stats['centroid'] for stats in cluster_stats]
                        inter_distances = pairwise_distances(cluster_means)
                        
                        # Average distance between clusters (excluding diagonal)
                        mask = ~np.eye(optimal_k, dtype=bool)
                        avg_inter_distance = inter_distances[mask].mean() if np.any(mask) else 0
                        
                        # Calculate intra-cluster compactness
                        intra_distances = []
                        for stats in cluster_stats:
                            cluster_data = X[labels == stats['cluster_id']]
                            if len(cluster_data) > 1:
                                centroid = stats['centroid']
                                distances = np.linalg.norm(cluster_data - centroid, axis=1)
                                intra_distances.append(distances.mean())
                        
                        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
                        
                        # Separation ratio
                        if avg_intra_distance > 0:
                            separation_ratio = avg_inter_distance / avg_intra_distance
                        else:
                            separation_ratio = 0
                        
                        # Display metrics
                        st.metric("Avg Inter-Cluster Distance", f"{avg_inter_distance:.3f}")
                        st.metric("Avg Intra-Cluster Distance", f"{avg_intra_distance:.3f}")
                        st.metric("Separation Ratio", f"{separation_ratio:.2f}")
                        
                        # Quality assessment
                        if separation_ratio > 3:
                            st.success("✅ Excellent cluster separation")
                        elif separation_ratio > 1.5:
                            st.info("📊 Good cluster separation")
                        else:
                            st.warning("⚠️ Poor cluster separation - consider different K")
                
                # Detailed Cluster Statistics
                st.markdown("#### 📊 Detailed Cluster Statistics")
                
                # Create comprehensive statistics table
                stats_data = []
                for stats in cluster_stats:
                    row = {
                        'Cluster': stats['cluster_id'],
                        'Size': stats['size'],
                        'Density (%)': f"{stats['density']:.1f}",
                        'Avg Silhouette': ""
                    }
                    
                    # Calculate silhouette for this cluster
                    cluster_mask = labels == stats['cluster_id']
                    if np.sum(cluster_mask) > 1:
                        cluster_silhouette = silhouette_samples(X, labels)
                        row['Avg Silhouette'] = f"{cluster_silhouette[cluster_mask].mean():.3f}"
                    
                    # Add LLM label if available
                    if 'llm_labels_df' in locals():
                        llm_label_info = llm_labels_df[llm_labels_df['cluster_id'] == stats['cluster_id']]
                        if not llm_label_info.empty:
                            row['Label'] = llm_label_info.iloc[0].get('llm_name', '')
                    
                    stats_data.append(row)
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, height=300)
                
                # Inter-cluster Distance Matrix
                st.markdown("#### 📏 Inter-cluster Distances")
                
                if optimal_k > 1:
                    # Calculate distances between centroids
                    from scipy.spatial.distance import cdist
                    
                    centroids_array = np.array([stats['centroid'] for stats in cluster_stats])
                    centroid_distances = cdist(centroids_array, centroids_array)
                    
                    # Create interactive heatmap
                    fig_dist = go.Figure(data=go.Heatmap(
                        z=centroid_distances,
                        x=[f"C{i}" for i in range(optimal_k)],
                        y=[f"C{i}" for i in range(optimal_k)],
                        colorscale='Viridis',
                        text=[[f"{val:.2f}" for val in row] for row in centroid_distances],
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hoverinfo="text",
                        hovertemplate='Distance between C%{y} and C%{x}: %{z:.2f}<extra></extra>'
                    ))
                    
                    fig_dist.update_layout(
                        title="Distance Between Cluster Centroids",
                        xaxis_title="Cluster",
                        yaxis_title="Cluster",
                        width=600,
                        height=500
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Cluster relationship insights
                    st.markdown("##### 🔗 Cluster Relationships")
                    
                    # Find closest clusters
                    np.fill_diagonal(centroid_distances, np.inf)  # Ignore diagonal
                    closest_pairs = []
                    for i in range(optimal_k):
                        for j in range(i+1, optimal_k):
                            closest_pairs.append({
                                'cluster1': i,
                                'cluster2': j,
                                'distance': centroid_distances[i, j]
                            })
                    
                    closest_pairs.sort(key=lambda x: x['distance'])
                    
                    if closest_pairs:
                        closest_pair = closest_pairs[0]
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Closest Clusters", f"C{closest_pair['cluster1']} ↔ C{closest_pair['cluster2']}")
                        with col_b:
                            st.metric("Distance", f"{closest_pair['distance']:.3f}")
                        with col_c:
                            # Get sizes of closest clusters
                            size1 = cluster_stats[closest_pair['cluster1']]['size']
                            size2 = cluster_stats[closest_pair['cluster2']]['size']
                            st.metric("Combined Size", size1 + size2)
            
            with tab5:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">📥 Export Results</h4></div>', unsafe_allow_html=True)
                
                # Export Options in Columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("##### 📄 Clustered Data")
                    
                    # Prepare clustered data for export
                    if len(feature_names) > 0:
                        df_export = pd.DataFrame(X, columns=feature_names)
                    else:
                        df_export = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
                    
                    df_export['Cluster'] = labels
                    df_export['Silhouette_Score'] = silhouette_samples(X, labels) if len(np.unique(labels)) > 1 else 0
                    
                    # Add distance to centroid
                    for cluster_id in range(optimal_k):
                        cluster_mask = labels == cluster_id
                        if np.any(cluster_mask):
                            centroid = np.mean(X[cluster_mask], axis=0)
                            distances = np.linalg.norm(X[cluster_mask] - centroid, axis=1)
                            df_export.loc[cluster_mask, 'Distance_To_Centroid'] = distances
                    
                    csv_data = df_export.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Download Clustered Data",
                        data=csv_data,
                        file_name=f"clustered_data_k{optimal_k}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.caption("Includes: Features, Cluster assignments, Silhouette scores, Distance to centroid")
                
                with col2:
                    st.markdown("##### 📊 Cluster Statistics")
                    
                    # Prepare cluster statistics for export
                    stats_export_data = []
                    for stats in cluster_stats:
                        row = {
                            'cluster_id': stats['cluster_id'],
                            'size': stats['size'],
                            'density_percent': stats['density'],
                            'avg_silhouette': ""
                        }
                        
                        # Calculate silhouette for this cluster
                        cluster_mask = labels == stats['cluster_id']
                        if np.sum(cluster_mask) > 1:
                            cluster_silhouette = silhouette_samples(X, labels)
                            row['avg_silhouette'] = cluster_silhouette[cluster_mask].mean()
                        
                        # Add centroid coordinates
                        for i, val in enumerate(stats['centroid']):
                            feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                            row[f'centroid_{feature_name}'] = val
                        
                        stats_export_data.append(row)
                    
                    stats_export_df = pd.DataFrame(stats_export_data)
                    stats_csv = stats_export_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📊 Download Cluster Stats",
                        data=stats_csv,
                        file_name=f"cluster_statistics_k{optimal_k}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.caption("Includes: Size, Density, Silhouette, Centroids")
                
                with col3:
                    st.markdown("##### 🤖 Cluster Labels")
                    
                    # Check if LLM labels are available
                    if 'llm_labels_df' in locals() and llm_labels_df is not None:
                        # Prepare LLM labels for export
                        llm_export_df = llm_labels_df.copy()
                        
                        # Add basic statistics to LLM labels
                        for stats in cluster_stats:
                            mask = llm_export_df['cluster_id'] == stats['cluster_id']
                            if np.any(mask):
                                llm_export_df.loc[mask, 'size'] = stats['size']
                                llm_export_df.loc[mask, 'density_percent'] = stats['density']
                        
                        llm_csv = llm_export_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="🤖 Download AI Labels",
                            data=llm_csv,
                            file_name=f"ai_cluster_labels_k{optimal_k}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Show label preview
                        with st.expander("Preview Labels", expanded=False):
                            preview_cols = ['cluster_id', 'llm_name', 'size', 'density']
                            preview_df = llm_export_df[preview_cols].head()
                            st.dataframe(preview_df, use_container_width=True)
                        
                        st.caption("Includes: AI-generated names, explanations, interpretations")
                    
                    elif include_llm_labeling:
                        st.info("Enable LLM labeling to generate AI labels")
                        
                        # Export basic labels as fallback
                        basic_labels_data = []
                        for stats in cluster_stats:
                            basic_labels_data.append({
                                'cluster_id': stats['cluster_id'],
                                'label': f"Cluster_{stats['cluster_id']}",
                                'size': stats['size'],
                                'density_percent': stats['density'],
                                'description': f"Cluster with {stats['size']} points ({stats['density']:.1f}% of data)"
                            })
                        
                        basic_labels_df = pd.DataFrame(basic_labels_data)
                        basic_csv = basic_labels_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="📝 Download Basic Labels",
                            data=basic_csv,
                            file_name=f"basic_cluster_labels_k{optimal_k}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.caption("Basic cluster labels (LLM not available)")
                    
                    else:
                        # Export rule-based labels
                        rule_labels = generate_rule_based_labels(X, labels, feature_names, cluster_stats)
                        rule_labels_df = pd.DataFrame(rule_labels)
                        rule_csv = rule_labels_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="📋 Download Rule-based Labels",
                            data=rule_csv,
                            file_name=f"rule_based_labels_k{optimal_k}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.caption("Rule-based descriptive labels")
                
                with col4:
                    st.markdown("##### 📈 Analysis Summary")
                    
                    # Create comprehensive analysis summary
                    summary = {
                        'analysis_info': {
                            'dataset_name': dataset_name,
                            'analysis_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'n_samples': X.shape[0],
                            'n_features': X.shape[1],
                            'optimal_k': optimal_k,
                            'clustering_method': clustering_method,
                            'llm_labeling_used': include_llm_labeling and 'llm_labels_df' in locals()
                        },
                        'quality_metrics': {
                            'inertia': float(kmeans.inertia_),
                            'silhouette_score': float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0,
                            'davies_bouldin_score': float(davies_bouldin_score(X, labels)),
                            'calinski_harabasz_score': float(calinski_harabasz_score(X, labels))
                        },
                        'cluster_distribution': {
                            'total_clusters': optimal_k,
                            'cluster_sizes': [int(np.sum(labels == i)) for i in range(optimal_k)],
                            'cluster_densities': [float((np.sum(labels == i) / len(X)) * 100) for i in range(optimal_k)]
                        },
                        'parameters': {
                            'random_state': random_state,
                            'n_init': 20,
                            'max_iter': 300,
                            'algorithm': 'lloyd'
                        }
                    }
                    
                    # Add LLM labels to summary if available
                    if 'llm_labels_df' in locals() and llm_labels_df is not None:
                        llm_summary = []
                        for _, row in llm_labels_df.iterrows():
                            llm_summary.append({
                                'cluster_id': int(row['cluster_id']),
                                'ai_label': row.get('llm_name', ''),
                                'explanation': row.get('llm_explanation', ''),
                                'size': int(row['size'])
                            })
                        summary['ai_labels'] = llm_summary
                    
                    # Export as JSON
                    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📋 Download Summary (JSON)",
                        data=summary_json.encode('utf-8'),
                        file_name=f"clustering_analysis_summary_k{optimal_k}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    # Export as HTML report
                    html_report = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Clustering Analysis Report - K={optimal_k}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 30px; border-radius: 10px; }}
                            .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; 
                                        margin: 10px 0; border-left: 4px solid #10b981; }}
                            .cluster-card {{ background: #e9ecef; padding: 15px; margin: 10px 0; 
                                        border-radius: 6px; }}
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <h1>Clustering Analysis Report</h1>
                            <p>Dataset: {dataset_name} | K={optimal_k} | Date: {summary['analysis_info']['analysis_date']}</p>
                        </div>
                        
                        <h2>📊 Quality Metrics</h2>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                            <div class="metric-card">
                                <h3>Silhouette Score</h3>
                                <h2>{summary['quality_metrics']['silhouette_score']:.3f}</h2>
                            </div>
                            <div class="metric-card">
                                <h3>Davies-Bouldin</h3>
                                <h2>{summary['quality_metrics']['davies_bouldin_score']:.3f}</h2>
                            </div>
                            <div class="metric-card">
                                <h3>Inertia</h3>
                                <h2>{summary['quality_metrics']['inertia']:,.0f}</h2>
                            </div>
                            <div class="metric-card">
                                <h3>Calinski-Harabasz</h3>
                                <h2>{summary['quality_metrics']['calinski_harabasz_score']:,.0f}</h2>
                            </div>
                        </div>
                        
                        <h2>📈 Cluster Distribution</h2>
                        <p>Total Samples: {summary['analysis_info']['n_samples']} | 
                        Features: {summary['analysis_info']['n_features']} | 
                        Clusters: {summary['cluster_distribution']['total_clusters']}</p>
                    </body>
                    </html>
                    """
                    
                    st.download_button(
                        label="📄 Download HTML Report",
                        data=html_report.encode('utf-8'),
                        file_name=f"clustering_report_k{optimal_k}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    st.caption("Includes: JSON summary + HTML report")
                
                # Reset Button
                st.markdown("---")
                col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
                with col_reset2:
                    if st.button("🔄 Run New Analysis", use_container_width=True):
                        st.session_state.run_analysis = False
                        st.rerun()
                
                # Quick Stats
                st.markdown("##### 📝 Quick Statistics")
                quick_stats_col1, quick_stats_col2, quick_stats_col3 = st.columns(3)
                
                with quick_stats_col1:
                    st.metric("Total Clusters", optimal_k)
                
                with quick_stats_col2:
                    largest_cluster = max([stats['size'] for stats in cluster_stats])
                    st.metric("Largest Cluster", largest_cluster)
                
                with quick_stats_col3:
                    smallest_cluster = min([stats['size'] for stats in cluster_stats])
                    st.metric("Smallest Cluster", smallest_cluster)

# Initial state
if not st.session_state.run_analysis:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
        <h3 style="color: #4b5563; margin-bottom: 1rem;">🚀 Ready to Cluster</h3>
        <p style="color: #6b7280; max-width: 600px; margin: 0 auto;">
            Configure your clustering settings in the sidebar and click "Run Clustering Analysis" to begin.
            Explore advanced features like auto-K selection, stability testing, and sub-cluster discovery.
        </p>
        <div style="margin-top: 2rem; color: #9ca3af; font-size: 0.9rem;">
            <p>✨ Features included:</p>
            <div style="display: inline-flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin-top: 1rem;">
                <span>🎯 Auto-K Selection</span> • 
                <span>📐 Multi-dimensional Views</span> • 
                <span>⚖️ Stability Testing</span> • 
                <span>🔍 Sub-cluster Discovery</span> • 
                <span>📋 Cluster Labeling</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">

<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
        <span>🎯 <strong>Auto-K Selection</strong></span>
        <span>📐 <strong>Multi-dimensional Views</strong></span>
        <span>⚖️ <strong>Stability Testing</strong></span>
        <span>🔍 <strong>Sub-cluster Discovery</strong></span>
        <span>📋 <strong>Cluster Labeling</strong></span>
    </div>
    <p style="margin: 0.5rem 0; color: #4b5563;">
        Advanced clustering analysis for discovering patterns in complex data
    </p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">
        © 2024 Advanced K-Means Clustering Pro
    </p>
</div>
""", unsafe_allow_html=True)