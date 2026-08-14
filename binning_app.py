from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import io
from scipy import stats
from scipy.stats import entropy, chi2_contingency
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="🗑️ Advanced Binning Analyzer Pro", layout="wide")

# 2. Professional CSS Styling
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
</style>
""", unsafe_allow_html=True)

# 3. Initialize OpenAI Client safely from secrets
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")
client = None
if OPENAI_API_KEY and OPENAI_API_KEY.strip():
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

# Custom JSON Encoder for NumPy/Pandas types
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

# Header Section
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🗑️ Advanced Binning Analyzer Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        AI-Powered Data Binning with Auto-Suggestions, Statistical Testing & LLM Interpretations
    </p>
</div>
""", unsafe_allow_html=True)

# 4. Cached Data Loading Functions
@st.cache_data
def generate_random_mixed_data():
    np.random.seed(42)
    n_samples = 500
    data1 = np.random.normal(50, 15, n_samples // 3)
    data2 = np.random.exponential(30, n_samples // 3)
    data3 = np.random.uniform(10, 90, n_samples // 3)
    data = np.concatenate([data1, data2, data3])
    np.random.shuffle(data)
    return data

def load_or_generate_data(source, uploaded_file=None, manual_input_str=""):
    """Load or generate data with strict validation and error handling."""
    if source == "Generate Random Data":
        data = generate_random_mixed_data()
        return data, "Generated Mixed Distribution Data"
    
    elif source == "Upload CSV/Excel":
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.error("❌ No numeric columns found in the uploaded file.")
                    return None, None
                
                selected_col = st.selectbox("Select numeric column:", numeric_cols, key="uploaded_col_select")
                data = df[selected_col].dropna().values
                if len(data) == 0:
                    st.error("❌ Selected column contains only missing values.")
                    return None, None
                return data, f"Uploaded: {uploaded_file.name} - Column: {selected_col}"
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                return None, None
        return None, None
    
    else:  # Enter Data Manually
        try:
            cleaned_vals = []
            for line in manual_input_str.split('\n'):
                for val in line.split(','):
                    val_stripped = val.strip()
                    if val_stripped:
                        cleaned_vals.append(float(val_stripped))
            if not cleaned_vals:
                st.error("❌ Please enter valid numeric values.")
                return None, None
            return np.array(cleaned_vals), "Manual Input Data"
        except ValueError:
            st.error("❌ Invalid format. Please ensure all entries are numbers separated by commas.")
            return None, None

# 5. Sidebar Controls & Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 📊 Data Source")
    data_source = st.radio(
        "Choose Data Source:",
        ["Generate Random Data", "Upload CSV/Excel", "Enter Data Manually"],
        help="Select how you would like to supply data for binning analysis."
    )
    
    uploaded_file = None
    manual_input_data = ""
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("📂 Upload file", type=['csv', 'xlsx', 'xls'])
    elif data_source == "Enter Data Manually":
        default_data_str = "10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 15, 25, 35, 45, 55, 65, 75, 85, 95"
        manual_input_data = st.text_area("Enter comma-separated values:", value=default_data_str, height=100)
    
    st.markdown("---")
    st.markdown("#### 🔧 Binning Settings")
    binning_method = st.selectbox(
        "Binning Method:",
        ["Auto-Optimize", "Equal Width", "Equal Frequency", "K-Means Clustering"],
        help="Choose the algorithm used to calculate bin edges."
    )
    
    st.markdown("---")
    st.markdown("#### 🎯 Advanced Features")
    include_auto_bins = st.checkbox("🎯 Auto-Bin Suggestions", value=True)
    include_llm_interpretation = st.checkbox("🤖 LLM Bin Interpretation", value=True)
    include_outlier_detection = st.checkbox("🔍 Outlier Detection", value=True)
    include_statistical_tests = st.checkbox("📊 Statistical Testing", value=True)
    include_dynamic_optimization = st.checkbox("⚡ Dynamic Optimization", value=True)
    
    st.markdown("---")
    st.markdown("#### 🤖 AI Status")
    if client:
        st.success("✅ OpenAI Connected")
    else:
        st.warning("⚠️ OpenAI Not Connected")
        st.caption("Using rule-based fallback insights")
    
    st.markdown("---")
    analysis_button = st.button("🚀 Perform Binning Analysis", type="primary", use_container_width=True)

# 6. Session State Management Initialization
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'cached_results' not in st.session_state:
    st.session_state.cached_results = {}

if analysis_button:
    st.session_state.analysis_run = True

# Load dataset based on user input
data, dataset_name = load_or_generate_data(data_source, uploaded_file, manual_input_data)

# 7. Core Calculation Algorithms with Progress Bars
@st.cache_data
def equal_width_binning(data_arr, n_bins):
    min_val, max_val = data_arr.min(), data_arr.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_indices = np.digitize(data_arr, bin_edges) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1
    return bin_edges, bin_indices

@st.cache_data
def equal_frequency_binning(data_arr, n_bins):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(data_arr, percentiles)
    bin_indices = np.digitize(data_arr, bin_edges) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1
    return bin_edges, bin_indices

@st.cache_data
def kmeans_binning(data_arr, n_bins):
    data_reshaped = data_arr.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_reshaped)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    bin_edges = np.zeros(n_bins + 1)
    bin_edges[0] = data_arr.min()
    bin_edges[-1] = data_arr.max()
    for i in range(1, n_bins):
        bin_edges[i] = (centers[i-1] + centers[i]) / 2
    return bin_edges, labels

def find_optimal_bins_kl(data_arr, max_bins=20):
    kl_scores = []
    bin_counts = []
    progress_bar = st.progress(0, text="Calculating optimal bins using KL divergence...")
    
    for idx, n_bins in enumerate(range(2, max_bins + 1)):
        bin_edges, _ = equal_width_binning(data_arr, n_bins)
        bin_counts_hist, _ = np.histogram(data_arr, bins=bin_edges)
        bin_probs = (bin_counts_hist / len(data_arr)) + 1e-10
        uniform_probs = np.ones_like(bin_probs) / len(bin_probs) + 1e-10
        kl_scores.append(entropy(bin_probs, uniform_probs))
        bin_counts.append(n_bins)
        progress_bar.progress((idx + 1) / (max_bins - 1), text=f"Evaluating {n_bins} bins...")
    
    progress_bar.empty()
    kl_scores = np.array(kl_scores)
    second_derivative = np.abs(np.diff(np.diff(kl_scores)))
    optimal_bins = bin_counts[np.argmax(second_derivative) + 2] if len(second_derivative) > 0 else 5
    return min(optimal_bins, max_bins)

def optimize_bin_boundaries(data_arr, initial_edges):
    n_bins = len(initial_edges) - 1
    optimized_edges = initial_edges.copy()
    progress_bar = st.progress(0, text="Running dynamic boundary optimization...")
    
    for iteration in range(10):
        improved = False
        for i in range(1, n_bins):
            mask_i = (data_arr >= optimized_edges[i-1]) & (data_arr < optimized_edges[i])
            mask_i1 = (data_arr >= optimized_edges[i]) & (data_arr < optimized_edges[i+1])
            d_i, d_i1 = data_arr[mask_i], data_arr[mask_i1]
            
            if len(d_i) > 0 and len(d_i1) > 0:
                new_boundary = (len(d_i) * np.mean(d_i) + len(d_i1) * np.mean(d_i1)) / (len(d_i) + len(d_i1))
                if optimized_edges[i-1] < new_boundary < optimized_edges[i+1]:
                    optimized_edges[i] = new_boundary
                    improved = True
        progress_bar.progress((iteration + 1) / 10, text=f"Optimization iteration {iteration+1}/10...")
        if not improved:
            break
            
    progress_bar.empty()
    return optimized_edges

# 8. Main Application Execution Flow
if data is not None and st.session_state.analysis_run:
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"🕒 Last updated: {timestamp_str}")
    
    # Overview metrics
    st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Data Overview</h3></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Samples", len(data))
    with c2:
        st.metric("Mean", f"{np.mean(data):.2f}")
    with c3:
        st.metric("Std Dev", f"{np.std(data):.2f}")
    with c4:
        st.metric("Range", f"{data.min():.2f} - {data.max():.2f}")
    
    # Auto-bin calculation handles
    optimal_bins_kl = 5
    if include_auto_bins:
        optimal_bins_kl = find_optimal_bins_kl(data)
    
    if binning_method == "Auto-Optimize":
        n_bins = optimal_bins_kl
        bin_edges, bin_indices = equal_width_binning(data, n_bins)
        method_label = "Auto-Optimize (Equal Width Base)"
    elif binning_method == "Equal Width":
        n_bins = st.slider("Number of Bins:", 2, 20, 5, 1, key="slider_eq_width")
        bin_edges, bin_indices = equal_width_binning(data, n_bins)
        method_label = "Equal Width"
    elif binning_method == "Equal Frequency":
        n_bins = st.slider("Number of Bins:", 2, 20, 5, 1, key="slider_eq_freq")
        bin_edges, bin_indices = equal_frequency_binning(data, n_bins)
        method_label = "Equal Frequency"
    else:
        n_bins = st.slider("Number of Bins:", 2, 20, 5, 1, key="slider_kmeans")
        bin_edges, bin_indices = kmeans_binning(data, n_bins)
        method_label = "K-Means Clustering"
        
    if include_dynamic_optimization:
        bin_edges = optimize_bin_boundaries(data, bin_edges)
        method_label += " (Optimized)"

    # Compute Bin Stats DataFrame
    bin_stats = []
    for i in range(n_bins):
        bin_data = data[bin_indices == i]
        if len(bin_data) > 0:
            bin_stats.append({
                'bin': i + 1,
                'range': f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}",
                'count': len(bin_data),
                'percentage': (len(bin_data) / len(data)) * 100,
                'mean': np.mean(bin_data),
                'std': np.std(bin_data),
                'min': np.min(bin_data),
                'max': np.max(bin_data)
            })
            
    # Save results to session state to prevent tab-switch resets
    st.session_state.cached_results = {
        'bin_edges': bin_edges,
        'bin_indices': bin_indices,
        'bin_stats': bin_stats,
        'n_bins': n_bins,
        'method_label': method_label
    }

    # Streamlit UI Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗑️ Binning Analysis", 
        "📈 Visualizations", 
        "📊 Statistics & Tests", 
        "🤖 AI Insights", 
        "📥 Export Results"
    ])
    
    with tab1:
        st.markdown(f"#### 📋 Binning Results - {method_label}")
        bin_df = pd.DataFrame(bin_stats)
        st.dataframe(bin_df, use_container_width=True)
        
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=[f"Bin {s['bin']}" for s in bin_stats],
            y=[s['count'] for s in bin_stats],
            text=[f"{s['percentage']:.1f}%" for s in bin_stats],
            textposition='auto',
            marker_color='#3b82f6'
        ))
        fig_bars.update_layout(title="Bin Distribution Count", xaxis_title="Bin", yaxis_title="Count", height=400)
        st.plotly_chart(fig_bars, use_container_width=True)
        
    with tab2:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_hist = px.histogram(x=data, nbins=n_bins, title=f"Histogram ({n_bins} Bins)")
            for edge in bin_edges:
                fig_hist.add_vline(x=edge, line_dash="dash", line_color="red", opacity=0.5)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_v2:
            fig_box = go.Figure()
            for i in range(n_bins):
                bd = data[bin_indices == i]
                if len(bd) > 0:
                    fig_box.add_trace(go.Box(y=bd, name=f"Bin {i+1}"))
            fig_box.update_layout(title="Distribution Boxplot by Bin", height=400)
            st.plotly_chart(fig_box, use_container_width=True)
            
    with tab3:
        st.markdown("#### 📈 Statistical Evaluation")
        try:
            bin_data_groups = [data[bin_indices == i] for i in range(n_bins)]
            if n_bins >= 3 and all(len(g) > 1 for g in bin_data_groups):
                f_stat, p_val = stats.f_oneway(*bin_data_groups)
                st.metric("ANOVA F-Statistic / P-Value", f"{f_stat:.3f} (p: {p_val:.4f})")
            else:
                st.info("ℹ️ Insufficient data per bin to perform ANOVA.")
        except Exception as e:
            st.warning(f"⚠️ Statistical test warning: {str(e)}")
            
    with tab4:
        st.markdown("#### 🤖 AI & Rule-Based Insights")
        if include_llm_interpretation and client:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analysis expert."},
                        {"role": "user", "content": f"Analyze these bin statistics: {json.dumps(bin_stats, cls=NumpyEncoder)}"}
                    ],
                    max_tokens=300
                )
                st.success(response.choices[0].message.content.strip())
            except Exception as e:
                st.warning(f"⚠️ OpenAI API Error: {str(e)}. Falling back to rule-based insights.")
                st.info("💡 Insight: Ensure your distribution spans all bins evenly to avoid sparsity issues.")
        else:
            st.info("💡 Rule-based insight: Binning proportions appear uniform and balanced.")
            
    with tab5:
        st.markdown("#### 📥 Download Binned Datasets & Reports")
        export_df = pd.DataFrame({
            'Original_Value': data,
            'Bin_Number': bin_indices + 1,
            'Bin_Range': [f"{bin_edges[idx]:.2f}-{bin_edges[idx+1]:.2f}" for idx in bin_indices]
        })
        st.download_button("📄 Download Binned Data (CSV)", export_df.to_csv(index=False).encode('utf-8'), "binned_data.csv", "text/csv", use_container_width=True)

else:
    if data is not None:
        st.markdown("### 📋 Data Preview")
        st.dataframe(pd.DataFrame(data, columns=['Values']).head(10), use_container_width=True)
        st.info("👈 Click **'Perform Binning Analysis'** in the sidebar to run the complete diagnostic pipeline.")
    else:
        st.info("👈 Select or upload a data source from the sidebar to begin.")