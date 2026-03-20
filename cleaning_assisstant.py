import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import re
from datetime import datetime
import openai
from faker import Faker
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# 🔥 HARDCODED OPENAI API KEY (Replace with your actual key)
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")

# 2. Initialize only if the key exists and isn't an empty string
if OPENAI_API_KEY and OPENAI_API_KEY.strip():
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Initialize Faker for synthetic data generation
fake = Faker()

# Set page config
st.set_page_config(page_title="🧹 Advanced Data Cleaning Assistant", layout="wide")

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

def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable types"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

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
    .warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    .info-card {
        background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🧹 Advanced Data Cleaning Assistant Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        AI-Powered Data Cleaning with Smart Imputation, Anomaly Detection & Privacy Protection
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 📊 Data Source")
    data_source = st.radio("Choose Data Source:", ["Use Sample Data", "Upload CSV/Excel"])
    
    # Advanced Features Toggles
    st.markdown("---")
    st.markdown("#### 🔧 Advanced Features")
    
    include_llm_anomaly = st.checkbox("🤖 LLM-Powered Anomaly Detection", value=True)
    include_smart_imputation = st.checkbox("🎯 Smart Imputation Suggestions", value=True)
    include_schema_inference = st.checkbox("🔍 Schema Inference", value=True)
    include_quality_report = st.checkbox("📊 Auto-Quality Report", value=True)
    include_privacy_transform = st.checkbox("🔒 Privacy Transformations", value=True)
    
    # OpenAI Status
    st.markdown("---")
    st.markdown("#### 🤖 AI Status")
    if client:
        st.success("✅ OpenAI Connected")
        st.caption("AI features enabled")
    else:
        st.warning("⚠️ OpenAI Not Connected")
        st.caption("Add API key in code (line 18)")
    
    st.markdown("---")
    
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

# Predefined Sample CSVs with Imperfections (Enhanced)
sample_data = {
    "Sales Data": pd.DataFrame({
        "Product_ID": ["P001", "P002", "P003", "P004", None, "P002", "P003"],
        "Product_Name": ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones", "Mouse", "Keyboard"],
        "Revenue": [1000, 1500, 2000, 2500, 3000, None, 2500],
        "Units_Sold": [10, 15, 20, 25, 30, 'fifteen', 25],
        "Sale_Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-02", "2024-01-03"],
        "Customer_Email": ["john@email.com", "alice@email.com", "bob@email.com", "eve@email.com", None, "alice@email.com", "bob@email.com"]
    }),
    
    "Employee Data": pd.DataFrame({
        "Employee_ID": ["E001", "E002", "E003", "E004", "E005", "E006"],
        "Employee_Name": ["John Smith", "Alice Johnson", "Bob Williams", "Eve Davis", "John Smith", None],
        "Salary_USD": [50000, 60000, 55000, 'not available', 70000, 60000],
        "Department": ["Human Resources", "Information Technology", None, "Marketing", "Human Resources", "Information Technology"],
        "Join_Date": ["2020-01-15", "2019-03-22", "2021-07-10", "invalid_date", "2020-01-15", "2022-11-30"],
        "SSN": ["123-45-6789", "987-65-4321", None, "555-55-5555", "123-45-6789", "987-65-4321"]
    }),
    
    "Customer Reviews": pd.DataFrame({
        "Review_ID": ["R001", "R002", "R003", "R004"],
        "Customer_Name": ["Tom Anderson", "Jerry Smith", None, "Donald Brown"],
        "Rating": [4.5, 3.8, 'five', 5.0],
        "Review_Text": ["Good product quality", None, "Nice experience", "Excellent service"],
        "Purchase_Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "Customer_Phone": ["+1-555-0101", "+1-555-0102", None, "+1-555-0104"]
    }),
    
    "Financial Transactions": pd.DataFrame({
        "Transaction_ID": ["T001", "T002", "T003", "T004"],
        "Date": ["2024-01-01", "2024-01-02", None, "2024-01-04"],
        "Amount": [150.50, 'high', 148.75, 155.00],
        "Account_Number": ["ACC123456", "ACC123456", "ACC789012", "ACC789012"],
        "Transaction_Type": ["Deposit", "Withdrawal", "Deposit", "Withdrawal"],
        "IP_Address": ["192.168.1.1", "10.0.0.1", None, "172.16.0.1"]
    })
}

# Initialize session state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

# Load data based on selection
if data_source == "Use Sample Data":
    selected_sample = st.selectbox("Choose a sample dataset:", list(sample_data.keys()))
    data = sample_data[selected_sample].copy()
    st.session_state.original_data = data.copy()
else:
    uploaded_file = st.file_uploader("📂 Upload your data file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.session_state.original_data = data.copy()
    else:
        data = None

# Main Analysis Execution
if data is not None and st.session_state.run_analysis:
    with st.spinner("🔍 Analyzing data quality and preparing cleaning recommendations..."):
        
        # Store original data
        original_data = data.copy()
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Overview", 
            "🔍 Quality Analysis", 
            "🤖 AI-Powered Cleaning",
            "🎯 Smart Imputation",
            "🔒 Privacy & Synthesis",
            "📤 Export Results"
        ])
        
        with tab1:
            st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Dataset Overview</h3></div>', unsafe_allow_html=True)
            
            # Quick Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", data.shape[0])
            with col2:
                st.metric("Total Columns", data.shape[1])
            with col3:
                missing_values = data.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            with col4:
                duplicate_rows = data.duplicated().sum()
                st.metric("Duplicate Rows", duplicate_rows)
            
            # Data Preview
            st.markdown("#### 📋 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Data Types
            st.markdown("#### 📝 Data Types")
            dtype_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum(),
                'Unique Values': [data[col].nunique() for col in data.columns]
            })
            st.dataframe(dtype_info, use_container_width=True)
        
        with tab2:
            st.markdown('<div class="section-header"><h3 style="margin: 0;">🔍 Data Quality Analysis</h3></div>', unsafe_allow_html=True)
            
            # Schema Inference
                        # Schema Inference
            if include_schema_inference:
                st.markdown("#### 🔍 Schema Inference & Validation")
                
                inferred_schema = {}
                for col in data.columns:
                    col_data = data[col].dropna()
                    
                    if len(col_data) > 0:
                        # Try to infer data type
                        sample_values = col_data.head(5).tolist()
                        
                        # Check for common patterns
                        col_str = str(col).lower()
                        
                        if any(keyword in col_str for keyword in ['date', 'time', 'year', 'month', 'day']):
                            inferred_type = 'datetime'
                            try:
                                pd.to_datetime(col_data, errors='coerce')
                                validation = "✅ Valid date format"
                            except:
                                validation = "⚠️ Invalid date format"
                        
                        elif any(keyword in col_str for keyword in ['email', 'mail']):
                            inferred_type = 'email'
                            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                            valid_emails = col_data.astype(str).str.match(email_pattern).sum()
                            validation = f"✅ {valid_emails}/{len(col_data)} valid emails"
                        
                        elif any(keyword in col_str for keyword in ['phone', 'tel', 'mobile']):
                            inferred_type = 'phone'
                            phone_pattern = r'^[\d\s\-\+\(\)]+$'
                            valid_phones = col_data.astype(str).str.match(phone_pattern).sum()
                            validation = f"✅ {valid_phones}/{len(col_data)} valid phones"
                        
                        elif any(keyword in col_str for keyword in ['id', 'code', 'sku']):
                            inferred_type = 'identifier'
                            validation = f"🔢 {col_data.nunique()} unique values"
                        
                        elif pd.api.types.is_numeric_dtype(col_data):
                            inferred_type = 'numeric'
                            validation = f"📊 Range: {col_data.min():.2f} to {col_data.max():.2f}"
                        
                        else:
                            inferred_type = 'text'
                            avg_length = col_data.astype(str).str.len().mean()
                            validation = f"📝 Avg length: {avg_length:.1f} chars"
                        
                        inferred_schema[col] = {
                            'inferred_type': inferred_type,
                            'validation': validation,
                            'sample_values': sample_values[:3]
                        }
                
                # Display inferred schema - FIXED VERSION
                if inferred_schema:
                    schema_data = []
                    for col, info in inferred_schema.items():
                        # Convert sample values to string representation
                        sample_vals = info['sample_values']
                        if sample_vals:
                            # Handle each sample value properly
                            sample_strs = []
                            for val in sample_vals:
                                if isinstance(val, (list, dict, tuple)):
                                    sample_strs.append(str(val))
                                else:
                                    sample_strs.append(str(val))
                            sample_display = ', '.join(sample_strs[:3])  # Show max 3
                        else:
                            sample_display = "No samples"
                        
                        schema_data.append({
                            'Column': col,
                            'Inferred Type': info['inferred_type'],
                            'Validation': info['validation'],
                            'Sample Values': sample_display
                        })
                    
                    schema_df = pd.DataFrame(schema_data)
                    st.dataframe(schema_df, use_container_width=True)
                else:
                    st.info("No schema inference available for empty dataset")
            # Data Quality Metrics
            st.markdown("#### 📊 Quality Metrics by Column")
            
            quality_metrics = []
            for col in data.columns:
                col_data = data[col]
                
                # Calculate metrics
                null_pct = (col_data.isnull().sum() / len(data)) * 100
                unique_pct = (col_data.nunique() / len(data)) * 100
                
                # Check for mixed types
                type_counts = col_data.apply(type).value_counts()
                is_mixed_type = len(type_counts) > 1
                
                # Check for outliers (for numeric columns)
                if pd.api.types.is_numeric_dtype(col_data):
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    outlier_count = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()
                else:
                    outlier_count = 0
                
                quality_metrics.append({
                    'Column': col,
                    'Data Type': str(col_data.dtype),
                    'Missing %': f"{null_pct:.1f}%",
                    'Unique %': f"{unique_pct:.1f}%",
                    'Mixed Types': "⚠️ Yes" if is_mixed_type else "✅ No",
                    'Outliers': outlier_count if outlier_count > 0 else "✅ None"
                })
            
            quality_df = pd.DataFrame(quality_metrics)
            st.dataframe(quality_df, use_container_width=True)
            
            # Visualization of data quality
            col1, col2 = st.columns(2)
            
            with col1:
                # Missing values heatmap
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(data.isnull(), cbar=False, cmap='viridis', ax=ax)
                ax.set_title('Missing Values Heatmap')
                st.pyplot(fig)
            
            with col2:
                # Data types distribution
                dtype_counts = data.dtypes.value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                dtype_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0'])
                ax.set_title('Data Types Distribution')
                ax.set_ylabel('')
                st.pyplot(fig)
        
        with tab3:
            st.markdown('<div class="section-header"><h3 style="margin: 0;">🤖 AI-Powered Data Cleaning</h3></div>', unsafe_allow_html=True)
            
            # LLM-Powered Anomaly Detection
            if include_llm_anomaly and client:
                st.markdown("#### 🤖 LLM-Powered Anomaly Detection")
                
                # Select column for anomaly analysis
                anomaly_col = st.selectbox("Select column for anomaly analysis:", data.columns)
                
                if st.button("🔍 Detect Anomalies with AI", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing anomalies..."):
                        try:
                            # Prepare sample data for LLM
                            sample_data_text = data[anomaly_col].dropna().head(20).tolist()
                            
                            # Call LLM for anomaly detection
                            prompt = f"""
                            Analyze this column data from a dataset and identify potential anomalies:
                            
                            Column Name: {anomaly_col}
                            Data Type: {data[anomaly_col].dtype}
                            Sample Values: {sample_data_text}
                            
                            Please identify:
                            1. Data type inconsistencies
                            2. Outliers or extreme values
                            3. Pattern violations
                            4. Suspicious entries
                            5. Recommended cleaning actions
                            
                            Format your response as a JSON with:
                            - anomalies_found: boolean
                            - anomaly_types: list of strings
                            - suspicious_values: list
                            - recommendations: list of strings
                            - confidence_score: float (0-1)
                            """
                            
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a data quality expert specializing in anomaly detection."},
                                    {"role": "user", "content": prompt}
                                ],
                                max_tokens=500,
                                temperature=0.3
                            )
                            
                            # Parse LLM response
                            llm_response = response.choices[0].message.content
                            
                            # Try to parse JSON
                            try:
                                anomaly_result = json.loads(llm_response)
                                
                                # Display results
                                if anomaly_result.get('anomalies_found', False):
                                    st.warning("⚠️ Anomalies Detected!")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.markdown("**Anomaly Types:**")
                                        for anomaly in anomaly_result.get('anomaly_types', []):
                                            st.caption(f"• {anomaly}")
                                    
                                    with col_b:
                                        st.markdown("**Suspicious Values:**")
                                        for value in anomaly_result.get('suspicious_values', [])[:5]:
                                            st.caption(f"• `{value}`")
                                    
                                    st.markdown("**Recommendations:**")
                                    for rec in anomaly_result.get('recommendations', []):
                                        st.info(f"📌 {rec}")
                                    
                                    # Apply automatic fixes
                                    if st.button("🔄 Apply Recommended Fixes", use_container_width=True):
                                        # Apply basic fixes based on recommendations
                                        for rec in anomaly_result.get('recommendations', []):
                                            if "convert to numeric" in rec.lower():
                                                try:
                                                    data[anomaly_col] = pd.to_numeric(data[anomaly_col], errors='coerce')
                                                    st.success(f"Converted {anomaly_col} to numeric")
                                                except:
                                                    pass
                                            elif "remove outliers" in rec.lower():
                                                # Simple outlier removal
                                                if pd.api.types.is_numeric_dtype(data[anomaly_col]):
                                                    q1 = data[anomaly_col].quantile(0.25)
                                                    q3 = data[anomaly_col].quantile(0.75)
                                                    iqr = q3 - q1
                                                    mask = (data[anomaly_col] >= (q1 - 1.5 * iqr)) & (data[anomaly_col] <= (q3 + 1.5 * iqr))
                                                    data = data[mask]
                                                    st.success(f"Removed outliers from {anomaly_col}")
                                else:
                                    st.success("✅ No anomalies detected by AI!")
                                
                            except json.JSONDecodeError:
                                st.info("AI Analysis Summary:")
                                st.write(llm_response)
                        
                        except Exception as e:
                            st.error(f"AI analysis failed: {str(e)}")
                            st.info("Using statistical anomaly detection instead...")
                
                # Statistical Anomaly Detection (fallback)
                st.markdown("#### 📊 Statistical Anomaly Detection")
                
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_num_col = st.selectbox("Select numeric column:", numeric_cols)
                    
                    if pd.api.types.is_numeric_dtype(data[selected_num_col]):
                        # Calculate outliers using IQR
                        q1 = data[selected_num_col].quantile(0.25)
                        q3 = data[selected_num_col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        outliers = data[(data[selected_num_col] < lower_bound) | (data[selected_num_col] > upper_bound)]
                        
                        if len(outliers) > 0:
                            st.warning(f"Found {len(outliers)} outliers in {selected_num_col}")
                            st.dataframe(outliers[[selected_num_col]], use_container_width=True)
                            
                            # Outlier treatment options
                            treatment = st.selectbox("Outlier Treatment:", 
                                                   ["Keep", "Remove", "Cap", "Impute with median"])
                            
                            if st.button("Apply Treatment", use_container_width=True):
                                if treatment == "Remove":
                                    data = data[(data[selected_num_col] >= lower_bound) & 
                                               (data[selected_num_col] <= upper_bound)]
                                    st.success("Outliers removed!")
                                elif treatment == "Cap":
                                    data[selected_num_col] = np.where(
                                        data[selected_num_col] < lower_bound, lower_bound,
                                        np.where(data[selected_num_col] > upper_bound, upper_bound, 
                                                data[selected_num_col])
                                    )
                                    st.success("Outliers capped!")
                                elif treatment == "Impute with median":
                                    median_val = data[selected_num_col].median()
                                    data.loc[(data[selected_num_col] < lower_bound) | 
                                            (data[selected_num_col] > upper_bound), selected_num_col] = median_val
                                    st.success("Outliers imputed with median!")
                        else:
                            st.success(f"✅ No statistical outliers found in {selected_num_col}")
            
            # Interactive Data Cleaning Tools
            st.markdown("#### 🛠️ Interactive Cleaning Tools")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Remove duplicates
                if st.button("🧹 Remove All Duplicates", use_container_width=True):
                    before = len(data)
                    data.drop_duplicates(inplace=True)
                    after = len(data)
                    st.success(f"Removed {before - after} duplicate rows")
            
            with col2:
                # Standardize text
                text_col = st.selectbox("Select text column to clean:", 
                                       data.select_dtypes(include=['object']).columns.tolist() + [None])
                if text_col and st.button("📝 Clean Text", use_container_width=True):
                    data[text_col] = data[text_col].astype(str).str.strip().str.title()
                    st.success(f"Cleaned {text_col}: Trimmed and title-cased")
        
        with tab4:
            st.markdown('<div class="section-header"><h3 style="margin: 0;">🎯 Smart Imputation & Transformation</h3></div>', unsafe_allow_html=True)
            
            if include_smart_imputation:
                st.markdown("#### 🎯 Smart Imputation Suggestions")
                
                # Identify columns with missing values
                missing_cols = data.columns[data.isnull().any()].tolist()
                
                if missing_cols:
                    st.info(f"Columns with missing values: {', '.join(missing_cols)}")
                    
                    for col in missing_cols:
                        with st.expander(f"Impute {col}", expanded=True):
                            col_data = data[col]
                            missing_count = col_data.isnull().sum()
                            total_count = len(col_data)
                            
                            st.caption(f"Missing: {missing_count}/{total_count} ({missing_count/total_count*100:.1f}%)")
                            
                            # Suggest imputation methods based on data type
                            if pd.api.types.is_numeric_dtype(col_data):
                                suggestions = [
                                    f"Mean: {col_data.mean():.2f}",
                                    f"Median: {col_data.median():.2f}",
                                    f"Forward fill",
                                    f"Backward fill",
                                    f"KNN imputation"
                                ]
                                default_method = "Median"
                            else:
                                suggestions = [
                                    f"Mode: {col_data.mode()[0] if not col_data.mode().empty else 'N/A'}",
                                    f"Forward fill",
                                    f"Backward fill",
                                    f"Custom value"
                                ]
                                default_method = "Mode"
                            
                            selected_method = st.selectbox(
                                f"Imputation method for {col}:",
                                suggestions,
                                key=f"impute_{col}"
                            )
                            
                            if st.button(f"Apply to {col}", key=f"apply_{col}", use_container_width=True):
                                if "Mean" in selected_method:
                                    impute_val = col_data.mean()
                                elif "Median" in selected_method:
                                    impute_val = col_data.median()
                                elif "Mode" in selected_method:
                                    impute_val = col_data.mode()[0] if not col_data.mode().empty else "Unknown"
                                elif "Forward" in selected_method:
                                    data[col] = col_data.ffill()
                                    st.success(f"Forward-filled {col}")
                                    continue
                                elif "Backward" in selected_method:
                                    data[col] = col_data.bfill()
                                    st.success(f"Backward-filled {col}")
                                    continue
                                elif "KNN" in selected_method:
                                    # Simple KNN imputation for numeric columns
                                    numeric_data = data.select_dtypes(include=[np.number])
                                    if len(numeric_data.columns) > 1:
                                        imputer = KNNImputer(n_neighbors=5)
                                        imputed = imputer.fit_transform(numeric_data)
                                        data[numeric_data.columns] = imputed
                                        st.success(f"KNN imputation applied to numeric columns")
                                    else:
                                        st.warning("Need at least 2 numeric columns for KNN")
                                    continue
                                else:
                                    impute_val = st.text_input(f"Custom value for {col}:", value="")
                                
                                if impute_val != "":
                                    data[col].fillna(impute_val, inplace=True)
                                    st.success(f"Imputed {col} with {impute_val}")
                else:
                    st.success("✅ No missing values found!")
            
            # Data Transformation
            st.markdown("#### 🔄 Data Transformations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Column type conversion
                convert_col = st.selectbox("Select column to convert:", data.columns)
                new_type = st.selectbox("Convert to:", 
                                       ['int', 'float', 'str', 'datetime', 'category'])
                
                if st.button("Convert Type", use_container_width=True):
                    try:
                        if new_type == 'datetime':
                            data[convert_col] = pd.to_datetime(data[convert_col], errors='coerce')
                        elif new_type == 'int':
                            data[convert_col] = pd.to_numeric(data[convert_col], errors='coerce').astype('Int64')
                        elif new_type == 'float':
                            data[convert_col] = pd.to_numeric(data[convert_col], errors='coerce')
                        elif new_type == 'category':
                            data[convert_col] = data[convert_col].astype('category')
                        else:
                            data[convert_col] = data[convert_col].astype(str)
                        
                        st.success(f"Converted {convert_col} to {new_type}")
                    except Exception as e:
                        st.error(f"Conversion failed: {str(e)}")
            
            with col2:
                # Standardization/Normalization
                num_cols = data.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    norm_col = st.selectbox("Select numeric column to normalize:", num_cols)
                    norm_method = st.selectbox("Normalization method:", 
                                             ['Min-Max Scaling', 'Z-Score', 'Log Transform'])
                    
                    if st.button("Apply Normalization", use_container_width=True):
                        if norm_method == 'Min-Max Scaling':
                            min_val = data[norm_col].min()
                            max_val = data[norm_col].max()
                            if max_val > min_val:
                                data[f"{norm_col}_normalized"] = (data[norm_col] - min_val) / (max_val - min_val)
                                st.success(f"Min-Max scaling applied to {norm_col}")
                        
                        elif norm_method == 'Z-Score':
                            mean_val = data[norm_col].mean()
                            std_val = data[norm_col].std()
                            if std_val > 0:
                                data[f"{norm_col}_zscore"] = (data[norm_col] - mean_val) / std_val
                                st.success(f"Z-Score normalization applied to {norm_col}")
        
        with tab5:
            st.markdown('<div class="section-header"><h3 style="margin: 0;">🔒 Privacy-Preserving Transformations</h3></div>', unsafe_allow_html=True)
            
            if include_privacy_transform:
                # Identify PII (Personally Identifiable Information) columns
                pii_patterns = {
                    'email': ['email', 'mail', 'e-mail'],
                    'phone': ['phone', 'tel', 'mobile', 'contact'],
                    'ssn': ['ssn', 'social', 'security'],
                    'address': ['address', 'street', 'city', 'zip', 'postal'],
                    'name': ['name', 'first', 'last', 'fullname'],
                    'credit_card': ['credit', 'card', 'cc', 'payment'],
                    'ip': ['ip', 'address_ip']
                }
                
                pii_columns = {}
                for col in data.columns:
                    col_lower = str(col).lower()
                    for pii_type, patterns in pii_patterns.items():
                        if any(pattern in col_lower for pattern in patterns):
                            if pii_type not in pii_columns:
                                pii_columns[pii_type] = []
                            pii_columns[pii_type].append(col)
                
                if pii_columns:
                    st.warning("⚠️ Potential PII Columns Detected:")
                    for pii_type, cols in pii_columns.items():
                        st.caption(f"**{pii_type.upper()}:** {', '.join(cols)}")
                    
                    # Privacy Transformation Options
                    st.markdown("#### 🛡️ Privacy Protection Options")
                    
                    selected_pii_type = st.selectbox("Select PII type to protect:", list(pii_columns.keys()))
                    selected_col = st.selectbox("Select column:", pii_columns[selected_pii_type])
                    
                    protection_method = st.selectbox(
                        "Protection method:",
                        ["Pseudonymization", "Masking", "Tokenization", "Synthetic Generation", "Aggregation"]
                    )
                    
                    if st.button("🔒 Apply Protection", use_container_width=True):
                        if protection_method == "Pseudonymization":
                            # Generate fake but realistic data
                            if selected_pii_type == 'email':
                                fake_data = [fake.email() for _ in range(len(data))]
                            elif selected_pii_type == 'phone':
                                fake_data = [fake.phone_number() for _ in range(len(data))]
                            elif selected_pii_type == 'name':
                                fake_data = [fake.name() for _ in range(len(data))]
                            elif selected_pii_type == 'address':
                                fake_data = [fake.address() for _ in range(len(data))]
                            else:
                                fake_data = [fake.uuid4() for _ in range(len(data))]
                            
                            data[f"{selected_col}_pseudonymized"] = fake_data
                            st.success(f"Pseudonymized {selected_col}")
                        
                        elif protection_method == "Masking":
                            # Mask sensitive parts
                            if selected_pii_type == 'email':
                                data[f"{selected_col}_masked"] = data[selected_col].apply(
                                    lambda x: re.sub(r'(@.*)', '@***.com', str(x)) if '@' in str(x) else x
                                )
                            elif selected_pii_type in ['phone', 'ssn', 'credit_card']:
                                data[f"{selected_col}_masked"] = data[selected_col].apply(
                                    lambda x: str(x)[:3] + '*' * (len(str(x)) - 6) + str(x)[-3:] if len(str(x)) > 6 else x
                                )
                            st.success(f"Masked {selected_col}")
                        
                        elif protection_method == "Synthetic Generation":
                            st.info("Generating synthetic dataset...")
                            # Create a synthetic version of the entire dataset
                            synthetic_data = data.copy()
                            
                            for col in data.columns:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    # Preserve distribution for numeric columns
                                    mean = data[col].mean()
                                    std = data[col].std()
                                    synthetic_data[col] = np.random.normal(mean, std, len(data))
                                elif data[col].nunique() < 20:
                                    # Preserve categories for categorical columns
                                    unique_vals = data[col].dropna().unique()
                                    synthetic_data[col] = np.random.choice(unique_vals, len(data))
                            
                            st.session_state.synthetic_data = synthetic_data
                            st.success("Synthetic dataset generated!")
                            st.dataframe(synthetic_data.head(), use_container_width=True)
                
                else:
                    st.success("✅ No obvious PII columns detected")
            
            # Data Quality Report Generation
                        # Data Quality Report Generation
            if include_quality_report:
                st.markdown("#### 📊 Auto-Quality Report Generation")
                
                if st.button("📈 Generate Quality Report", use_container_width=True):
                    with st.spinner("Generating comprehensive quality report..."):
                        # Calculate comprehensive metrics - convert all to native Python types
                        report = {
                            "timestamp": datetime.now().isoformat(),
                            "dataset_info": {
                                "rows": int(data.shape[0]),
                                "columns": int(data.shape[1]),
                                "memory_usage": float(data.memory_usage(deep=True).sum() / 1024 / 1024)
                            },
                            "quality_metrics": {
                                "completeness": float((1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100),
                                "uniqueness": float((data.nunique().sum() / (data.shape[0] * data.shape[1])) * 100),
                                "duplicate_rows": int(data.duplicated().sum()),
                                "mixed_types": int(sum([len(data[col].apply(type).value_counts()) > 1 for col in data.columns]))
                            },
                            "column_analysis": {},
                            "recommendations": []
                        }
                        
                        # Column-level analysis - convert all values to native Python types
                        for col in data.columns:
                            col_data = data[col]
                            report["column_analysis"][col] = {
                                "data_type": str(col_data.dtype),
                                "missing_values": int(col_data.isnull().sum()),
                                "missing_percentage": float((col_data.isnull().sum() / len(data)) * 100),
                                "unique_values": int(col_data.nunique()),
                                "sample_values": col_data.dropna().head(3).astype(str).tolist()  # Convert to string list
                            }
                        
                        # Generate recommendations
                        if report["quality_metrics"]["duplicate_rows"] > 0:
                            report["recommendations"].append("Remove duplicate rows to improve data quality")
                        
                        if report["quality_metrics"]["mixed_types"] > 0:
                            report["recommendations"].append("Fix mixed data types in columns")
                        
                        if report["quality_metrics"]["completeness"] < 90:
                            report["recommendations"].append("Consider imputation for missing values")
                        
                        # Display report
                        st.markdown("##### 📋 Quality Report Summary")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Completeness", f"{report['quality_metrics']['completeness']:.1f}%")
                        with col_b:
                            st.metric("Uniqueness", f"{report['quality_metrics']['uniqueness']:.1f}%")
                        with col_c:
                            st.metric("Duplicate Rows", report['quality_metrics']['duplicate_rows'])
                        
                        # Download report using custom encoder
                        report_json = json.dumps(report, indent=2, cls=NumpyEncoder, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download Quality Report (JSON)",
                            data=report_json.encode('utf-8'),
                            file_name="data_quality_report.json",
                            mime="application/json",
                            use_container_width=True
                        )
            
        with tab6:
            st.markdown('<div class="section-header"><h3 style="margin: 0;">📤 Export Results</h3></div>', unsafe_allow_html=True)
            
            # Cleaned Data Preview
            st.markdown("#### 📝 Cleaned Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Export Options
            st.markdown("#### 📤 Export Options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Export as CSV
                csv_data = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export as Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name='Cleaned Data')
                    if 'original_data' in st.session_state:
                        st.session_state.original_data.to_excel(writer, index=False, sheet_name='Original Data')
                
                st.download_button(
                    label="📊 Download Excel",
                    data=output.getvalue(),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                # Export synthetic data if generated
                if 'synthetic_data' in st.session_state:
                    synthetic_csv = st.session_state.synthetic_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🔒 Download Synthetic Data",
                        data=synthetic_csv,
                        file_name="synthetic_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("Generate synthetic data in Privacy tab")
            
            with col4:
                # Export cleaning log
                cleaning_log = {
                    "cleaning_timestamp": datetime.now().isoformat(),
                    "original_shape": {
                        "rows": int(st.session_state.original_data.shape[0]) if st.session_state.original_data is not None else None,
                        "columns": int(st.session_state.original_data.shape[1]) if st.session_state.original_data is not None else None
                    },
                    "cleaned_shape": {
                        "rows": int(data.shape[0]),
                        "columns": int(data.shape[1])
                    },
                    "rows_removed": int(st.session_state.original_data.shape[0] - data.shape[0]) if st.session_state.original_data is not None else 0,
                    "columns_changed": list(data.columns),
                    "missing_values_before": int(st.session_state.original_data.isnull().sum().sum()) if st.session_state.original_data is not None else 0,
                    "missing_values_after": int(data.isnull().sum().sum()),
                    "data_types_summary": {
                        col: str(data[col].dtype) for col in data.columns
                    },
                    "cleaning_operations": [
                        "Schema inference",
                        "Missing value analysis",
                        "Anomaly detection" if include_llm_anomaly else None,
                        "Smart imputation" if include_smart_imputation else None,
                        "Privacy transformations" if include_privacy_transform else None
                    ]
                }
                
                # Remove None values from cleaning operations
                cleaning_log["cleaning_operations"] = [op for op in cleaning_log["cleaning_operations"] if op is not None]
                
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
                        else:
                            return super(NumpyEncoder, self).default(obj)
                
                log_json = json.dumps(cleaning_log, indent=2, cls=NumpyEncoder)
                
                st.download_button(
                    label="📋 Download Cleaning Log",
                    data=log_json.encode('utf-8'),
                    file_name="cleaning_log.json",
                    mime="application/json",
                    use_container_width=True
                )
            # Comparison Metrics
            st.markdown("#### 📊 Before & After Comparison")
            
            if st.session_state.original_data is not None:
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    before_missing = st.session_state.original_data.isnull().sum().sum()
                    after_missing = data.isnull().sum().sum()
                    st.metric("Missing Values", f"{after_missing}", f"{-before_missing + after_missing}")
                
                with col_b:
                    before_dupes = st.session_state.original_data.duplicated().sum()
                    after_dupes = data.duplicated().sum()
                    st.metric("Duplicate Rows", f"{after_dupes}", f"{-before_dupes + after_dupes}")
                
                with col_c:
                    before_cols = len(st.session_state.original_data.columns)
                    after_cols = len(data.columns)
                    st.metric("Total Columns", f"{after_cols}", f"{after_cols - before_cols}")
            
            # Reset Button
            st.markdown("---")
            if st.button("🔄 Start New Analysis", use_container_width=True):
                st.session_state.run_analysis = False
                st.rerun()
    
    # Store cleaned data
    st.session_state.cleaned_data = data

else:
    if data is not None:
        # Show data preview before analysis
        st.markdown("### 📋 Data Preview")
        st.dataframe(data.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", data.shape[0])
            st.metric("Total Columns", data.shape[1])
        with col2:
            st.metric("Missing Values", data.isnull().sum().sum())
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        st.info("Click 'Start Analysis' in the sidebar to begin advanced data cleaning.")
    else:
        st.info("👈 Select a data source to begin analysis.")

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">

<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
        <span>🤖 <strong>LLM Anomaly Detection</strong></span>
        <span>🎯 <strong>Smart Imputation</strong></span>
        <span>🔍 <strong>Schema Inference</strong></span>
        <span>📊 <strong>Auto-Quality Reports</strong></span>
        <span>🔒 <strong>Privacy Protection</strong></span>
    </div>
    <p style="margin: 0.5rem 0; color: #4b5563;">
        Advanced data cleaning with AI-powered insights and privacy protection
    </p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">
        © 2024 Advanced Data Cleaning Assistant Pro
    </p>
</div>
""", unsafe_allow_html=True)