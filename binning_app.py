from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
from scipy import stats
from scipy.stats import entropy, kstest, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import openai
import warnings
warnings.filterwarnings('ignore')

# 1. Fetch the key from secrets
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")

# 2. Initialize only if the key exists and isn't an empty string
if OPENAI_API_KEY and OPENAI_API_KEY.strip():
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
        else:
            return super(NumpyEncoder, self).default(obj)

# Set page config
st.set_page_config(page_title="🗑️ Advanced Binning Analyzer", layout="wide")

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
    .bin-card {
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
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🗑️ Advanced Binning Analyzer Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        AI-Powered Data Binning with Auto-Suggestions, Statistical Testing & LLM Interpretations
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 📊 Data Source")
    data_source = st.radio(
        "Choose Data Source:",
        ["Generate Random Data", "Upload CSV/Excel", "Enter Data Manually"]
    )
    
    st.markdown("---")
    st.markdown("#### 🔧 Binning Settings")
    
    binning_method = st.selectbox(
        "Binning Method:",
        ["Auto-Optimize", "Equal Width", "Equal Frequency", "K-Means Clustering", "Custom"]
    )
    
    # Advanced Features Toggles
    st.markdown("---")
    st.markdown("#### 🎯 Advanced Features")
    
    include_auto_bins = st.checkbox("🎯 Auto-Bin Suggestions", value=True)
    include_llm_interpretation = st.checkbox("🤖 LLM Bin Interpretation", value=True)
    include_outlier_detection = st.checkbox("🔍 Outlier Detection", value=True)
    include_statistical_tests = st.checkbox("📊 Statistical Testing", value=True)
    include_dynamic_optimization = st.checkbox("⚡ Dynamic Optimization", value=True)
    
    # OpenAI Status
    st.markdown("---")
    st.markdown("#### 🤖 AI Status")
    if client:
        st.success("✅ OpenAI Connected")
        st.caption("LLM interpretations enabled")
    else:
        st.warning("⚠️ OpenAI Not Connected")
        st.caption("Add API key in code (line 23)")
    
    st.markdown("---")
    
    if st.button("🚀 Perform Binning Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

# Initialize session state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Data Generation and Loading Functions
def load_data(source):
    """Load data based on selected source"""
    if source == "Generate Random Data":
        # Generate realistic random data with different distributions
        np.random.seed(42)
        
        # Mixed distribution data
        n_samples = 500
        data1 = np.random.normal(50, 15, n_samples//3)
        data2 = np.random.exponential(30, n_samples//3)
        data3 = np.random.uniform(10, 90, n_samples//3)
        
        data = np.concatenate([data1, data2, data3])
        np.random.shuffle(data)
        
        return data, "Generated Mixed Distribution Data"
    
    elif source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("📂 Upload your data file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Select numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select numeric column:", numeric_cols)
                data = df[selected_col].dropna().values
                return data, f"Uploaded: {uploaded_file.name} - Column: {selected_col}"
            else:
                st.error("❌ No numeric columns found in the uploaded file.")
                return None, None
        
        return None, None
    
    else:  # Enter Data Manually
        default_data = """10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                        15, 25, 35, 45, 55, 65, 75, 85, 95,
                        12, 22, 32, 42, 52, 62, 72, 82, 92"""
        
        user_data = st.text_area("Enter comma-separated numeric values:", 
                                value=default_data, height=100)
        
        try:
            data = []
            for line in user_data.split('\n'):
                for val in line.split(','):
                    val = val.strip()
                    if val:
                        data.append(float(val))
            return np.array(data), "Manual Input Data"
        except:
            st.error("❌ Invalid data format. Please enter comma-separated numbers.")
            return None, None

# Binning Functions
def equal_width_binning(data, n_bins):
    """Equal width binning"""
    min_val, max_val = data.min(), data.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_indices = np.digitize(data, bin_edges) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1  # Handle edge case
    
    return bin_edges, bin_indices

def equal_frequency_binning(data, n_bins):
    """Equal frequency binning"""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(data, percentiles)
    bin_indices = np.digitize(data, bin_edges) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1
    
    return bin_edges, bin_indices

def kmeans_binning(data, n_bins):
    """K-Means clustering for binning"""
    data_reshaped = data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_reshaped)
    
    # Get bin edges from cluster centers
    centers = np.sort(kmeans.cluster_centers_.flatten())
    bin_edges = np.zeros(n_bins + 1)
    bin_edges[0] = data.min()
    bin_edges[-1] = data.max()
    
    for i in range(1, n_bins):
        bin_edges[i] = (centers[i-1] + centers[i]) / 2
    
    return bin_edges, labels

# Auto-Bin Optimization Functions
def calculate_kl_divergence(data, bin_edges):
    """Calculate KL divergence between bin distribution and uniform distribution"""
    bin_counts, _ = np.histogram(data, bins=bin_edges)
    bin_probs = bin_counts / len(data)
    
    # Uniform distribution for comparison
    uniform_probs = np.ones_like(bin_probs) / len(bin_probs)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    bin_probs = bin_probs + epsilon
    uniform_probs = uniform_probs + epsilon
    
    # Calculate KL divergence
    kl_div = entropy(bin_probs, uniform_probs)
    
    return kl_div

def find_optimal_bins_kl(data, max_bins=20):
    """Find optimal number of bins using KL divergence"""
    kl_scores = []
    bin_counts = []
    
    for n_bins in range(2, max_bins + 1):
        bin_edges, _ = equal_width_binning(data, n_bins)
        kl_div = calculate_kl_divergence(data, bin_edges)
        kl_scores.append(kl_div)
        bin_counts.append(n_bins)
    
    # Find elbow point (maximum curvature)
    kl_scores = np.array(kl_scores)
    second_derivative = np.abs(np.diff(np.diff(kl_scores)))
    
    if len(second_derivative) > 0:
        optimal_idx = np.argmax(second_derivative) + 2
        optimal_bins = bin_counts[optimal_idx]
    else:
        optimal_bins = 5  # Default
    
    return min(optimal_bins, max_bins), kl_scores, bin_counts

def find_optimal_bins_sturges(data):
    """Sturges' formula for optimal bin count"""
    n = len(data)
    optimal = int(np.ceil(np.log2(n)) + 1)
    return max(2, min(optimal, 20))

def find_optimal_bins_scott(data):
    """Scott's normal reference rule"""
    sigma = np.std(data)
    n = len(data)
    optimal = int(np.ceil((data.max() - data.min()) / (3.5 * sigma / (n ** (1/3)))))
    return max(2, min(optimal, 20))

# Outlier Detection Functions
def detect_outliers_iqr(data):
    """Detect outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data))
    outliers = data[z_scores > threshold]
    return outliers, z_scores

# Statistical Testing Functions
def perform_bin_statistical_tests(data, bin_edges, bin_indices):
    """Perform statistical tests for bins"""
    results = {}
    n_bins = len(bin_edges) - 1
    
    # Prepare data for tests
    bin_data = [data[bin_indices == i] for i in range(n_bins)]
    
    # ANOVA test (if at least 3 bins with sufficient data)
    if n_bins >= 3 and all(len(bd) > 1 for bd in bin_data):
        try:
            f_stat, p_value = stats.f_oneway(*bin_data)
            results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        except:
            results['anova'] = {'error': 'ANOVA test failed'}
    
    # Kruskal-Wallis test (non-parametric alternative)
    if n_bins >= 3 and all(len(bd) > 0 for bd in bin_data):
        try:
            h_stat, p_value = stats.kruskal(*bin_data)
            results['kruskal_wallis'] = {
                'h_statistic': float(h_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        except:
            results['kruskal_wallis'] = {'error': 'Kruskal-Wallis test failed'}
    
    # Chi-square test for uniformity
    bin_counts = np.array([len(bd) for bd in bin_data])
    expected_counts = np.full(n_bins, len(data) / n_bins)
    
    try:
        chi2_stat, p_value, _, _ = chi2_contingency([bin_counts, expected_counts])
        results['chi_square_uniformity'] = {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    except:
        results['chi_square_uniformity'] = {'error': 'Chi-square test failed'}
    
    return results

# LLM Interpretation Functions
def generate_llm_bin_interpretation(client, data, bin_edges, bin_stats, dataset_name):
    """Generate LLM interpretation for binning results"""
    if not client:
        return None
    
    try:
        # Prepare bin information for LLM
        bin_info = []
        for i, stats in enumerate(bin_stats):
            bin_info.append(f"""
            Bin {i+1} ({bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}):
            - Count: {stats['count']} samples
            - Percentage: {stats['percentage']:.1f}%
            - Mean: {stats['mean']:.2f}
            - Std Dev: {stats['std']:.2f}
            - Min: {stats['min']:.2f}
            - Max: {stats['max']:.2f}
            """)
        
        prompt = f"""
        I have performed data binning analysis on a dataset. Here are the results:
        
        Dataset: {dataset_name}
        Total Samples: {len(data)}
        Number of Bins: {len(bin_stats)}
        
        Bin Statistics:
        {''.join(bin_info)}
        
        Please provide:
        1. **Overall Interpretation**: What does this binning tell us about the data distribution?
        2. **Key Insights**: What are the most important observations from the bin statistics?
        3. **Recommendations**: What actions or further analyses would you suggest?
        4. **Business Implications**: How could these bins be used in practical applications?
        
        Keep the response concise and actionable.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analysis expert who provides insightful interpretations of binning results."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"LLM interpretation failed: {str(e)}")
        return None

# Dynamic Optimization Functions
def optimize_bin_boundaries(data, initial_bin_edges):
    """Optimize bin boundaries to improve information gain"""
    n_bins = len(initial_bin_edges) - 1
    optimized_edges = initial_bin_edges.copy()
    
    for iteration in range(10):  # Maximum iterations
        improved = False
        
        for i in range(1, n_bins):
            # Get data for adjacent bins
            bin_i_mask = (data >= optimized_edges[i-1]) & (data < optimized_edges[i])
            bin_i1_mask = (data >= optimized_edges[i]) & (data < optimized_edges[i+1])
            
            bin_i_data = data[bin_i_mask]
            bin_i1_data = data[bin_i1_mask]
            
            if len(bin_i_data) > 0 and len(bin_i1_data) > 0:
                # Calculate current separation
                current_sep = optimized_edges[i]
                
                # Try moving boundary based on distribution
                mean_i = np.mean(bin_i_data)
                mean_i1 = np.mean(bin_i1_data)
                
                # Weighted average for new boundary
                n_i = len(bin_i_data)
                n_i1 = len(bin_i1_data)
                new_boundary = (n_i * mean_i + n_i1 * mean_i1) / (n_i + n_i1)
                
                # Ensure boundary stays between bins
                if optimized_edges[i-1] < new_boundary < optimized_edges[i+1]:
                    optimized_edges[i] = new_boundary
                    improved = True
        
        if not improved:
            break
    
    return optimized_edges

# Main Analysis Execution
data, dataset_name = load_data(data_source)

if data is not None and st.session_state.run_analysis:
    with st.spinner("🔍 Performing advanced binning analysis..."):
        
        # Data Overview
        st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Data Overview</h3></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Mean", f"{np.mean(data):.2f}")
        with col3:
            st.metric("Std Dev", f"{np.std(data):.2f}")
        with col4:
            st.metric("Range", f"{data.min():.2f} - {data.max():.2f}")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗑️ Binning Analysis", 
            "📈 Visualizations", 
            "📊 Statistics & Tests",
            "🤖 AI Insights",
            "📥 Export Results"
        ])
        
        with tab1:
            st.markdown('<div class="section-header"><h4 style="margin: 0;">🗑️ Binning Configuration</h4></div>', unsafe_allow_html=True)
            
            # Auto-bin suggestion
            if include_auto_bins:
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    sturges_bins = find_optimal_bins_sturges(data)
                    st.metric("Sturges Formula", sturges_bins)
                
                with col_b:
                    scott_bins = find_optimal_bins_scott(data)
                    st.metric("Scott's Rule", scott_bins)
                
                with col_c:
                    optimal_bins_kl, kl_scores, bin_counts = find_optimal_bins_kl(data)
                    st.metric("KL Divergence", optimal_bins_kl)
            
            # Bin count selection
            if binning_method == "Auto-Optimize":
                n_bins = optimal_bins_kl if include_auto_bins else 5
                st.info(f"📊 Auto-selected {n_bins} bins")
            else:
                n_bins = st.slider("Number of Bins:", 2, 20, 5, 1)
            
            # Perform binning based on selected method
            if binning_method == "Equal Width":
                bin_edges, bin_indices = equal_width_binning(data, n_bins)
                method_name = "Equal Width"
            elif binning_method == "Equal Frequency":
                bin_edges, bin_indices = equal_frequency_binning(data, n_bins)
                method_name = "Equal Frequency"
            elif binning_method == "K-Means Clustering":
                bin_edges, bin_indices = kmeans_binning(data, n_bins)
                method_name = "K-Means Clustering"
            else:  # Auto-Optimize or Custom
                bin_edges, bin_indices = equal_width_binning(data, n_bins)
                method_name = "Equal Width"
            
            # Dynamic optimization
            if include_dynamic_optimization:
                bin_edges = optimize_bin_boundaries(data, bin_edges)
                method_name += " (Optimized)"
            
            # Display bin results
            st.markdown(f"#### 📋 Binning Results - {method_name}")
            
            # Calculate bin statistics
            bin_stats = []
            for i in range(n_bins):
                bin_data = data[bin_indices == i]
                if len(bin_data) > 0:
                    stats_dict = {
                        'bin': i + 1,
                        'range': f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}",
                        'count': len(bin_data),
                        'percentage': (len(bin_data) / len(data)) * 100,
                        'mean': np.mean(bin_data),
                        'std': np.std(bin_data),
                        'min': np.min(bin_data),
                        'max': np.max(bin_data)
                    }
                    bin_stats.append(stats_dict)
            
            # Display bin statistics table
            bin_df = pd.DataFrame(bin_stats)
            st.dataframe(bin_df, use_container_width=True)
            
            # Bin distribution visualization
            fig_bars = go.Figure()
            fig_bars.add_trace(go.Bar(
                x=[f"Bin {i+1}" for i in range(n_bins)],
                y=[stats['count'] for stats in bin_stats],
                text=[f"{stats['percentage']:.1f}%" for stats in bin_stats],
                textposition='auto',
                marker_color='#3b82f6'
            ))
            
            fig_bars.update_layout(
                title='Bin Distribution',
                xaxis_title='Bin',
                yaxis_title='Count',
                height=400
            )
            
            st.plotly_chart(fig_bars, use_container_width=True)
        
        with tab2:
            st.markdown('<div class="section-header"><h4 style="margin: 0;">📈 Data Visualizations</h4></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram with bin edges
                fig_hist = px.histogram(
                    x=data,
                    nbins=n_bins,
                    title=f'Histogram with {n_bins} Bins',
                    labels={'x': 'Value', 'y': 'Frequency'},
                    color_discrete_sequence=['#3b82f6']
                )
                
                # Add bin edges as vertical lines
                for edge in bin_edges:
                    fig_hist.add_vline(
                        x=edge, 
                        line_dash="dash", 
                        line_color="red",
                        opacity=0.5
                    )
                
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot by bin
                fig_box = go.Figure()
                
                for i in range(n_bins):
                    bin_data = data[bin_indices == i]
                    if len(bin_data) > 0:
                        fig_box.add_trace(go.Box(
                            y=bin_data,
                            name=f'Bin {i+1}',
                            boxpoints='outliers'
                        ))
                
                fig_box.update_layout(
                    title='Distribution by Bin',
                    yaxis_title='Value',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Outlier detection visualization
            if include_outlier_detection:
                st.markdown("#### 🔍 Outlier Analysis")
                
                outliers_iqr, lower_bound, upper_bound = detect_outliers_iqr(data)
                outliers_zscore, z_scores = detect_outliers_zscore(data)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("IQR Outliers", len(outliers_iqr))
                with col_b:
                    st.metric("Z-Score Outliers", len(outliers_zscore))
                with col_c:
                    outlier_pct = (len(outliers_iqr) / len(data)) * 100
                    st.metric("Outlier Percentage", f"{outlier_pct:.1f}%")
                
                # Scatter plot with outliers highlighted
                fig_outliers = go.Figure()
                
                # Normal points
                normal_mask = (data >= lower_bound) & (data <= upper_bound)
                fig_outliers.add_trace(go.Scatter(
                    x=np.arange(len(data))[normal_mask],
                    y=data[normal_mask],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=8, opacity=0.6)
                ))
                
                # Outliers
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                fig_outliers.add_trace(go.Scatter(
                    x=np.arange(len(data))[outlier_mask],
                    y=data[outlier_mask],
                    mode='markers',
                    name='Outliers',
                    marker=dict(color='red', size=10, symbol='x')
                ))
                
                # Add bounds
                fig_outliers.add_hline(
                    y=lower_bound, 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text=f"Lower Bound: {lower_bound:.2f}"
                )
                
                fig_outliers.add_hline(
                    y=upper_bound, 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text=f"Upper Bound: {upper_bound:.2f}"
                )
                
                fig_outliers.update_layout(
                    title='Outlier Detection (IQR Method)',
                    xaxis_title='Index',
                    yaxis_title='Value',
                    height=400
                )
                
                st.plotly_chart(fig_outliers, use_container_width=True)
        
        with tab3:
            st.markdown('<div class="section-header"><h4 style="margin: 0;">📊 Statistical Analysis</h4></div>', unsafe_allow_html=True)
            
            # Statistical tests
            if include_statistical_tests:
                test_results = perform_bin_statistical_tests(data, bin_edges, bin_indices)
                
                # Display test results
                if 'anova' in test_results:
                    st.markdown("##### 📈 ANOVA Test")
                    anova = test_results['anova']
                    if 'error' not in anova:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("F-Statistic", f"{anova['f_statistic']:.3f}")
                        with col_b:
                            st.metric("P-Value", f"{anova['p_value']:.3f}")
                        with col_c:
                            if anova['significant']:
                                st.success("✅ Significant differences")
                            else:
                                st.info("⚠️ No significant differences")
                
                if 'kruskal_wallis' in test_results:
                    st.markdown("##### 📊 Kruskal-Wallis Test")
                    kw = test_results['kruskal_wallis']
                    if 'error' not in kw:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("H-Statistic", f"{kw['h_statistic']:.3f}")
                        with col_b:
                            st.metric("P-Value", f"{kw['p_value']:.3f}")
                        with col_c:
                            if kw['significant']:
                                st.success("✅ Significant differences")
                            else:
                                st.info("⚠️ No significant differences")
                
                # Bin comparison matrix
                st.markdown("##### 🔗 Bin Comparison Matrix")
                
                # Create correlation-like matrix
                comparison_data = []
                for i in range(n_bins):
                    row = []
                    for j in range(n_bins):
                        if i == j:
                            row.append(1.0)
                        else:
                            # Calculate overlap percentage
                            bin_i_data = data[bin_indices == i]
                            bin_j_data = data[bin_indices == j]
                            
                            if len(bin_i_data) > 0 and len(bin_j_data) > 0:
                                # Simple distance measure
                                mean_i = np.mean(bin_i_data)
                                mean_j = np.mean(bin_j_data)
                                std_i = np.std(bin_i_data)
                                std_j = np.std(bin_j_data)
                                
                                # Normalized distance
                                if std_i + std_j > 0:
                                    distance = abs(mean_i - mean_j) / (std_i + std_j)
                                    similarity = 1 / (1 + distance)
                                else:
                                    similarity = 0
                            else:
                                similarity = 0
                            
                            row.append(similarity)
                    comparison_data.append(row)
                
                fig_matrix = go.Figure(data=go.Heatmap(
                    z=comparison_data,
                    x=[f"Bin {i+1}" for i in range(n_bins)],
                    y=[f"Bin {i+1}" for i in range(n_bins)],
                    colorscale='RdBu',
                    text=[[f"{val:.2f}" for val in row] for row in comparison_data],
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                
                fig_matrix.update_layout(
                    title='Bin Similarity Matrix',
                    height=400
                )
                
                st.plotly_chart(fig_matrix, use_container_width=True)
            
            # Data distribution statistics
            st.markdown("##### 📊 Distribution Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                skewness = stats.skew(data)
                st.metric("Skewness", f"{skewness:.3f}")
                if abs(skewness) > 1:
                    st.caption("Highly skewed")
                elif abs(skewness) > 0.5:
                    st.caption("Moderately skewed")
                else:
                    st.caption("Approximately symmetric")
            
            with col2:
                kurtosis = stats.kurtosis(data)
                st.metric("Kurtosis", f"{kurtosis:.3f}")
                if kurtosis > 3:
                    st.caption("Leptokurtic (heavy tails)")
                elif kurtosis < 3:
                    st.caption("Platykurtic (light tails)")
                else:
                    st.caption("Mesokurtic (normal-like)")
            
            with col3:
                # Normality test
                _, p_norm = stats.shapiro(data) if len(data) < 5000 else stats.normaltest(data)
                st.metric("Normality P-Value", f"{p_norm:.4f}")
                if p_norm < 0.05:
                    st.caption("Not normally distributed")
                else:
                    st.caption("Normally distributed")
        
        with tab4:
            st.markdown('<div class="section-header"><h4 style="margin: 0;">🤖 AI Insights & Interpretations</h4></div>', unsafe_allow_html=True)
            
            if include_llm_interpretation and client:
                if st.button("🤖 Generate AI Interpretation", use_container_width=True):
                    with st.spinner("Generating AI insights..."):
                        interpretation = generate_llm_bin_interpretation(
                            client, data, bin_edges, bin_stats, dataset_name
                        )
                        
                        if interpretation:
                            st.success("✅ AI Interpretation Generated")
                            st.markdown("#### 📝 AI Analysis")
                            st.markdown(interpretation)
                        else:
                            st.warning("AI interpretation failed. Check API key and connection.")
            
            elif include_llm_interpretation:
                st.info("⚠️ OpenAI API key required for LLM interpretations")
            
            # Rule-based insights
            st.markdown("#### 📊 Data Insights")
            
            # Generate insights from bin statistics
            insights = []
            
            # Check for uneven distribution
            bin_counts = [stats['count'] for stats in bin_stats]
            max_bin = max(bin_counts)
            min_bin = min(bin_counts)
            
            if max_bin > 3 * min_bin:
                insights.append("⚠️ **Uneven bin distribution** - Some bins have significantly more data than others")
            
            # Check for large standard deviations
            large_std_bins = [i+1 for i, stats in enumerate(bin_stats) 
                            if stats['std'] > np.std(data)]
            if large_std_bins:
                insights.append(f"🔍 **High variability** in bins: {', '.join(map(str, large_std_bins))}")
            
            # Check for overlapping ranges
            for i in range(len(bin_stats)-1):
                if bin_stats[i]['max'] > bin_stats[i+1]['min']:
                    insights.append(f"🔄 **Overlap detected** between Bin {i+1} and Bin {i+2}")
                    break
            
            # Display insights
            if insights:
                for insight in insights:
                    with st.container():
                        st.info(insight)
            else:
                st.success("✅ Binning appears well-structured")
            
            # Recommendations
            st.markdown("#### 🎯 Recommendations")
            
            recommendations = []
            if n_bins < 5:
                recommendations.append("Consider increasing bin count for finer granularity")
            elif n_bins > 10:
                recommendations.append("Consider reducing bin count to avoid over-segmentation")
            
            if include_outlier_detection and len(outliers_iqr) > 0.1 * len(data):
                recommendations.append("Investigate and potentially handle outliers")
            
            if include_dynamic_optimization:
                recommendations.append("Dynamic optimization applied for better bin boundaries")
            
            for rec in recommendations:
                with st.container():
                    st.caption(f"• {rec}")
        
        with tab5:
            st.markdown('<div class="section-header"><h4 style="margin: 0;">📥 Export Results</h4></div>', unsafe_allow_html=True)
            
            # Export Options
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Export binned data
                export_df = pd.DataFrame({
                    'Original_Value': data,
                    'Bin_Number': bin_indices + 1,
                    'Bin_Range': [f"{bin_edges[bin_idx]:.2f}-{bin_edges[bin_idx+1]:.2f}" 
                                for bin_idx in bin_indices]
                })
                
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Binned Data (CSV)",
                    data=csv_data,
                    file_name="binned_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export bin statistics
                stats_df = pd.DataFrame(bin_stats)
                stats_csv = stats_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Bin Statistics (CSV)",
                    data=stats_csv,
                    file_name="bin_statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Export analysis summary
                summary = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'dataset_name': dataset_name,
                    'dataset_size': len(data),
                    'binning_method': method_name,
                    'number_of_bins': n_bins,
                    'data_statistics': {
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'skewness': float(stats.skew(data)),
                        'kurtosis': float(stats.kurtosis(data))
                    },
                    'bin_summary': {
                        'total_samples': len(data),
                        'average_bin_size': float(len(data) / n_bins),
                        'bin_variability': float(np.std([s['count'] for s in bin_stats]) / np.mean([s['count'] for s in bin_stats]))
                    }
                }
                
                summary_json = json.dumps(summary, indent=2, cls=NumpyEncoder)
                st.download_button(
                    label="📋 Analysis Summary (JSON)",
                    data=summary_json.encode('utf-8'),
                    file_name="binning_analysis_summary.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col4:
                # Export visualizations
                if st.button("🖼️ Save Visualizations", use_container_width=True):
                    # Save histogram
                    fig_hist.write_image("histogram.png")
                    st.success("Visualizations saved to disk")
            
            # Reset Button
            st.markdown("---")
            if st.button("🔄 Start New Analysis", use_container_width=True):
                st.session_state.run_analysis = False
                st.rerun()
    
else:
    if data is not None:
        # Show data preview before analysis
        st.markdown("### 📋 Data Preview")
        
        preview_df = pd.DataFrame(data, columns=['Values'])
        st.dataframe(preview_df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(data))
        with col2:
            st.metric("Mean", f"{np.mean(data):.2f}")
        with col3:
            st.metric("Range", f"{data.min():.2f} - {data.max():.2f}")
        
        st.info("Click 'Perform Binning Analysis' in the sidebar to begin advanced analysis.")
    else:
        st.info("👈 Select a data source to begin analysis.")

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">

<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
        <span>🎯 <strong>Auto-Bin Suggestions</strong></span>
        <span>🤖 <strong>LLM Interpretations</strong></span>
        <span>⚡ <strong>Dynamic Optimization</strong></span>
        <span>🔍 <strong>Outlier Detection</strong></span>
        <span>📊 <strong>Statistical Testing</strong></span>
    </div>
    <p style="margin: 0.5rem 0; color: #4b5563;">
        Advanced data binning with AI-powered insights and statistical validation
    </p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">
        © 2024 Advanced Binning Analyzer Pro
    </p>
</div>
""", unsafe_allow_html=True)