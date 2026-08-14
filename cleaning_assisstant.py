from datetime import datetime
import io
import json
import re
import warnings
from faker import Faker
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import seaborn as sns
import streamlit as st

warnings.filterwarnings('ignore')

# 🔥 HARDCODED OPENAI API KEY (Replace with your actual key)
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")

# 2. Initialize only if the key exists and isn't an empty string
if OPENAI_API_KEY and OPENAI_API_KEY.strip():
  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
  except Exception:
    client = None
else:
  client = None

# Initialize Faker for synthetic data generation
fake = Faker()

# Set page config
st.set_page_config(
    page_title="🧹 Advanced Data Cleaning Assistant", layout="wide"
)

# Configuration Dictionary (Basic Improvement: Centralize Constants & Config)
CONFIG = {
    "max_file_size_mb": 200,
    "default_model": "gpt-3.5-turbo",
    "pii_patterns": {
        "email": ["email", "mail", "e-mail"],
        "phone": ["phone", "tel", "mobile", "contact"],
        "ssn": ["ssn", "social", "security"],
        "address": ["address", "street", "city", "zip", "postal"],
        "name": ["name", "first", "last", "fullname"],
        "credit_card": ["credit", "card", "cc", "payment"],
        "ip": ["ip", "address_ip"],
    },
}

# Consolidated NumpyEncoder (Removed Duplicate Code)


class NumpyEncoder(json.JSONEncoder):
  """Custom JSON encoder for NumPy and Pandas types."""

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
      return obj.to_dict(orient="records")
    elif hasattr(obj, "item"):
      return obj.item()
    elif pd.isna(obj):
      return None
    else:
      return super(NumpyEncoder, self).default(obj)


# Professional CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🧹 Advanced Data Cleaning Assistant Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        AI-Powered Data Cleaning with Smart Imputation, Anomaly Detection & Privacy Protection
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# Cached Data Loading Functions (Basic Improvement: Caching & Performance)
@st.cache_data
def load_sample_dataset(dataset_name: str) -> pd.DataFrame:
  """Load pre-defined sample datasets with caching."""
  sample_data = {
      "Sales Data": pd.DataFrame({
          "Product_ID": [
              "P001",
              "P002",
              "P003",
              "P004",
              None,
              "P002",
              "P003",
          ],
          "Product_Name": [
              "Laptop",
              "Mouse",
              "Keyboard",
              "Monitor",
              "Headphones",
              "Mouse",
              "Keyboard",
          ],
          "Revenue": [1000, 1500, 2000, 2500, 3000, None, 2500],
          "Units_Sold": [10, 15, 20, 25, 30, "fifteen", 25],
          "Sale_Date": [
              "2024-01-01",
              "2024-01-02",
              "2024-01-03",
              "2024-01-04",
              "2024-01-05",
              "2024-01-02",
              "2024-01-03",
          ],
          "Customer_Email": [
              "john@email.com",
              "alice@email.com",
              "bob@email.com",
              "eve@email.com",
              None,
              "alice@email.com",
              "bob@email.com",
          ],
      }),
      "Employee Data": pd.DataFrame({
          "Employee_ID": ["E001", "E002", "E003", "E004", "E005", "E006"],
          "Employee_Name": [
              "John Smith",
              "Alice Johnson",
              "Bob Williams",
              "Eve Davis",
              "John Smith",
              None,
          ],
          "Salary_USD": [50000, 60000, 55000, "not available", 70000, 60000],
          "Department": [
              "Human Resources",
              "Information Technology",
              None,
              "Marketing",
              "Human Resources",
              "Information Technology",
          ],
          "Join_Date": [
              "2020-01-15",
              "2019-03-22",
              "2021-07-10",
              "invalid_date",
              "2020-01-15",
              "2022-11-30",
          ],
          "SSN": [
              "123-45-6789",
              "987-65-4321",
              None,
              "555-55-5555",
              "123-45-6789",
              "987-65-4321",
          ],
      }),
      "Customer Reviews": pd.DataFrame({
          "Review_ID": ["R001", "R002", "R003", "R004"],
          "Customer_Name": ["Tom Anderson", "Jerry Smith", None, "Donald Brown"],
          "Rating": [4.5, 3.8, "five", 5.0],
          "Review_Text": [
              "Good product quality",
              None,
              "Nice experience",
              "Excellent service",
          ],
          "Purchase_Date": [
              "2024-01-01",
              "2024-01-02",
              "2024-01-03",
              "2024-01-04",
          ],
          "Customer_Phone": ["+1-555-0101", "+1-555-0102", None, "+1-555-0104"],
      }),
      "Financial Transactions": pd.DataFrame({
          "Transaction_ID": ["T001", "T002", "T003", "T004"],
          "Date": ["2024-01-01", "2024-01-02", None, "2024-01-04"],
          "Amount": [150.50, "high", 148.75, 155.00],
          "Account_Number": [
              "ACC123456",
              "ACC123456",
              "ACC789012",
              "ACC789012",
          ],
          "Transaction_Type": ["Deposit", "Withdrawal", "Deposit", "Withdrawal"],
          "IP_Address": ["192.168.1.1", "10.0.0.1", None, "172.16.0.1"],
      }),
  }
  return sample_data.get(dataset_name, sample_data["Sales Data"])


@st.cache_data
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
  """Load uploaded CSV or Excel file safely with caching."""
  if uploaded_file.name.endswith(".csv"):
    return pd.read_csv(uploaded_file)
  else:
    return pd.read_excel(uploaded_file)


# Sidebar Configuration with improved tooltips and organized options
with st.sidebar:
  st.markdown("### ⚙️ Configuration")

  st.markdown("#### 📊 Data Source")
  data_source = st.radio(
      "Choose Data Source:",
      ["Use Sample Data", "Upload CSV/Excel"],
      help="Select whether to use built-in sample datasets or upload your own file.",
  )

  # Advanced Features Toggles with Enhanced Tooltips
  st.markdown("---")
  st.markdown("#### 🔧 Advanced Features")

  include_llm_anomaly = st.checkbox(
      "🤖 LLM-Powered Anomaly Detection",
      value=True,
      help="Leverage OpenAI to find complex anomalies and textual inconsistencies.",
  )
  include_smart_imputation = st.checkbox(
      "🎯 Smart Imputation Suggestions",
      value=True,
      help="Get intelligent suggestions for filling missing data points.",
  )
  include_schema_inference = st.checkbox(
      "🔍 Schema Inference",
      value=True,
      help="Automatically infer patterns like dates, emails, and phone numbers.",
  )
  include_quality_report = st.checkbox(
      "📊 Auto-Quality Report",
      value=True,
      help="Generate an exportable comprehensive data health report.",
  )
  include_privacy_transform = st.checkbox(
      "🔒 Privacy Transformations",
      value=True,
      help="Detect PII and apply masking or pseudonymization.",
  )

  # OpenAI Status
  st.markdown("---")
  st.markdown("#### 🤖 AI Status")
  if client:
    st.success("✅ OpenAI Connected")
    st.caption("AI features enabled")
  else:
    st.warning("⚠️ OpenAI Not Connected")
    st.caption("Add API key in secrets or environment variables")

  st.markdown("---")

  # Improved Reset & Responsive Button Behavior
  col_btn1, col_btn2 = st.columns(2)
  with col_btn1:
    analysis_button = st.button(
        "🚀 Start Analysis",
        type="primary",
        use_container_width=True,
        help="Click to run cleaning and diagnostics pipeline.",
    )
  with col_btn2:
    if st.button(
        "🔄 Reset",
        use_container_width=True,
        help="Clear all session state history and restart app.",
    ):
      for key in list(st.session_state.keys()):
        del st.session_state[key]
      st.rerun()

if "run_analysis" not in st.session_state:
  st.session_state.run_analysis = False
if "original_data" not in st.session_state:
  st.session_state.original_data = None
if "cleaned_data" not in st.session_state:
  st.session_state.cleaned_data = None
if "synthetic_data" not in st.session_state:
  st.session_state.synthetic_data = None

if analysis_button:
  st.session_state.run_analysis = True

# Data Loading with Basic Validation and Error Handling
df = None
if data_source == "Use Sample Data":
  selected_sample = st.selectbox(
      "Choose a sample dataset:",
      ["Sales Data", "Employee Data", "Customer Reviews", "Financial Transactions"],
  )
  df = load_sample_dataset(selected_sample)
  st.session_state.original_data = df.copy()
else:
  uploaded_file = st.file_uploader(
      "📂 Upload your data file",
      type=["csv", "xlsx", "xls"],
      help="Upload CSV or Excel files under 200MB.",
  )
  if uploaded_file is not None:
    # Validate file size (<200MB)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > CONFIG["max_file_size_mb"]:
      st.error(
          f"❌ File size exceeds limit ({file_size_mb:.1f}MB >"
          f" {CONFIG['max_file_size_mb']}MB)."
      )
    else:
      try:
        df = load_uploaded_file(uploaded_file)
        st.session_state.original_data = df.copy()
      except Exception as e:
        st.error(
            f"❌ Failed to parse file. Please verify file format. Details: {e}"
        )

# Main Analysis Execution Flow
if df is not None and st.session_state.run_analysis:
  # Progress bar integration for long running diagnostics (Basic Improvement)
  progress_bar = st.progress(0, text="Initializing data processing pipeline...")

  with st.spinner(
      "🔍 Analyzing data quality and preparing cleaning recommendations..."
  ):
    progress_bar.progress(20, text="Loading dataset attributes...")
    working_df = df.copy()
    progress_bar.progress(40, text="Computing quality metrics...")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🔍 Quality Analysis",
        "🤖 AI-Powered Cleaning",
        "🎯 Smart Imputation",
        "🔒 Privacy & Synthesis",
        "📤 Export Results",
    ])

    progress_bar.progress(100, text="Analysis pipeline ready!")
    progress_bar.empty()

    with tab1:
      st.markdown(
          '<div class="section-header"><h3 style="margin: 0;">📊 Dataset'
          " Overview</h3></div>",
          unsafe_allow_html=True,
      )

      # Quick Stats
      col1, col2, col3, col4 = st.columns(4)
      with col1:
        st.metric("Total Rows", working_df.shape[0])
      with col2:
        st.metric("Total Columns", working_df.shape[1])
      with col3:
        missing_values = working_df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
      with col4:
        duplicate_rows = working_df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_rows)

      # Data Preview
      st.markdown("#### 📋 Data Preview")
      st.dataframe(working_df.head(10), use_container_width=True)

      # Data Types
      st.markdown("#### 📝 Data Types")
      dtype_info = pd.DataFrame({
          "Column": working_df.columns,
          "Data Type": working_df.dtypes.astype(str),
          "Non-Null Count": working_df.count(),
          "Null Count": working_df.isnull().sum(),
          "Unique Values": [working_df[col].nunique() for col in working_df.columns],
      })
      st.dataframe(dtype_info, use_container_width=True)

    with tab2:
      st.markdown(
          '<div class="section-header"><h3 style="margin: 0;">🔍 Data Quality'
          " Analysis</h3></div>",
          unsafe_allow_html=True,
      )

      # Schema Inference
      if include_schema_inference:
        st.markdown("#### 🔍 Schema Inference & Validation")

        inferred_schema = {}
        for col in working_df.columns:
          col_data = working_df[col].dropna()

          if len(col_data) > 0:
            sample_values = col_data.head(5).tolist()
            col_str = str(col).lower()

            if any(
                keyword in col_str
                for keyword in ["date", "time", "year", "month", "day"]
            ):
              inferred_type = "datetime"
              try:
                pd.to_datetime(col_data, errors="coerce")
                validation = "✅ Valid date format"
              except Exception:
                validation = "⚠️ Invalid date format"

            elif any(keyword in col_str for keyword in ["email", "mail"]):
              inferred_type = "email"
              email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
              valid_emails = (
                  col_data.astype(str).str.match(email_pattern).sum()
              )
              validation = f"✅ {valid_emails}/{len(col_data)} valid emails"

            elif any(
                keyword in col_str for keyword in ["phone", "tel", "mobile"]
            ):
              inferred_type = "phone"
              phone_pattern = r"^[\d\s\-\+\(\)]+$"
              valid_phones = (
                  col_data.astype(str).str.match(phone_pattern).sum()
              )
              validation = f"✅ {valid_phones}/{len(col_data)} valid phones"

            elif any(keyword in col_str for keyword in ["id", "code", "sku"]):
              inferred_type = "identifier"
              validation = f"🔢 {col_data.nunique()} unique values"

            elif pd.api.types.is_numeric_dtype(col_data):
              inferred_type = "numeric"
              validation = f"📊 Range: {col_data.min():.2f} to {col_data.max():.2f}"

            else:
              inferred_type = "text"
              avg_length = col_data.astype(str).str.len().mean()
              validation = f"📝 Avg length: {avg_length:.1f} chars"

            inferred_schema[col] = {
                "inferred_type": inferred_type,
                "validation": validation,
                "sample_values": sample_values[:3],
            }

        if inferred_schema:
          schema_data = []
          for col, info in inferred_schema.items():
            sample_vals = info["sample_values"]
            if sample_vals:
              sample_strs = [str(val) for val in sample_vals]
              sample_display = ", ".join(sample_strs[:3])
            else:
              sample_display = "No samples"

            schema_data.append({
                "Column": col,
                "Inferred Type": info["inferred_type"],
                "Validation": info["validation"],
                "Sample Values": sample_display,
            })

          schema_df = pd.DataFrame(schema_data)
          st.dataframe(schema_df, use_container_width=True)
        else:
          st.info("No schema inference available for empty dataset")

      # Data Quality Metrics
      st.markdown("#### 📊 Quality Metrics by Column")

      quality_metrics = []
      for col in working_df.columns:
        col_data = working_df[col]
        null_pct = (col_data.isnull().sum() / len(working_df)) * 100
        unique_pct = (col_data.nunique() / len(working_df)) * 100
        type_counts = col_data.apply(type).value_counts()
        is_mixed_type = len(type_counts) > 1

        if pd.api.types.is_numeric_dtype(col_data):
          q1 = col_data.quantile(0.25)
          q3 = col_data.quantile(0.75)
          iqr = q3 - q1
          outlier_count = (
              (col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))
          ).sum()
        else:
          outlier_count = 0

        quality_metrics.append({
            "Column": col,
            "Data Type": str(col_data.dtype),
            "Missing %": f"{null_pct:.1f}%",
            "Unique %": f"{unique_pct:.1f}%",
            "Mixed Types": "⚠️ Yes" if is_mixed_type else "✅ No",
            "Outliers": outlier_count if outlier_count > 0 else "✅ None",
        })

      quality_df = pd.DataFrame(quality_metrics)
      st.dataframe(quality_df, use_container_width=True)

      col1, col2 = st.columns(2)
      with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(working_df.isnull(), cbar=False, cmap="viridis", ax=ax)
        ax.set_title("Missing Values Heatmap")
        st.pyplot(fig)

      with col2:
        dtype_counts = working_df.dtypes.value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        dtype_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax,
            colors=["#FF6B6B", "#4ECDC4", "#FFD166", "#06D6A0"],
        )
        ax.set_title("Data Types Distribution")
        ax.set_ylabel("")
        st.pyplot(fig)

    with tab3:
      st.markdown(
          '<div class="section-header"><h3 style="margin: 0;">🤖 AI-Powered Data'
          " Cleaning</h3></div>",
          unsafe_allow_html=True,
      )

      if include_llm_anomaly and client:
        st.markdown("#### 🤖 LLM-Powered Anomaly Detection")
        anomaly_col = st.selectbox(
            "Select column for anomaly analysis:", working_df.columns
        )

        if st.button("🔍 Detect Anomalies with AI", use_container_width=True):
          with st.spinner("🤖 AI is analyzing anomalies..."):
            try:
              sample_data_text = (
                  working_df[anomaly_col].dropna().head(20).tolist()
              )
              prompt = f"""
                            Analyze this column data from a dataset and identify potential anomalies:
                            
                            Column Name: {anomaly_col}
                            Data Type: {working_df[anomaly_col].dtype}
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
                  model=CONFIG["default_model"],
                  messages=[
                      {
                          "role": "system",
                          "content": (
                              "You are a data quality expert specializing in"
                              " anomaly detection. Always return valid JSON."
                          ),
                      },
                      {"role": "user", "content": prompt},
                  ],
                  max_tokens=500,
                  temperature=0.3,
              )

              llm_response = response.choices[0].message.content

              # Robust JSON parsing with fallback (Basic Improvement)
              try:
                # Strip code blocks if present in response
                cleaned_json_str = re.sub(
                    r"^```json\s*|\s*```$", "", llm_response.strip()
                )
                anomaly_result = json.loads(cleaned_json_str)

                if anomaly_result.get("anomalies_found", False):
                  st.warning("⚠️ Anomalies Detected!")

                  col_a, col_b = st.columns(2)
                  with col_a:
                    st.markdown("**Anomaly Types:**")
                    for anomaly in anomaly_result.get("anomaly_types", []):
                      st.caption(f"• {anomaly}")

                  with col_b:
                    st.markdown("**Suspicious Values:**")
                    for value in anomaly_result.get(
                        "suspicious_values", []
                    )[:5]:
                      st.caption(f"• `{value}`")

                  st.markdown("**Recommendations:**")
                  for rec in anomaly_result.get("recommendations", []):
                    st.info(f"📌 {rec}")

                  if st.button(
                      "🔄 Apply Recommended Fixes", use_container_width=True
                  ):
                    for rec in anomaly_result.get("recommendations", []):
                      if "convert to numeric" in rec.lower():
                        try:
                          working_df[anomaly_col] = pd.to_numeric(
                              working_df[anomaly_col], errors="coerce"
                          )
                          st.success(f"Converted {anomaly_col} to numeric")
                        except Exception:
                          pass
                      elif "remove outliers" in rec.lower():
                        if pd.api.types.is_numeric_dtype(
                            working_df[anomaly_col]
                        ):
                          q1 = working_df[anomaly_col].quantile(0.25)
                          q3 = working_df[anomaly_col].quantile(0.75)
                          iqr = q3 - q1
                          mask = (
                              working_df[anomaly_col] >= (q1 - 1.5 * iqr)
                          ) & (working_df[anomaly_col] <= (q3 + 1.5 * iqr))
                          working_df = working_df[mask]
                          st.success(f"Removed outliers from {anomaly_col}")
                else:
                  st.success("✅ No anomalies detected by AI!")

              except json.JSONDecodeError:
                st.warning(
                    "⚠️ Could not parse LLM output as JSON. Displaying raw"
                    " output below:"
                )
                st.info(llm_response)

            except Exception as e:
              st.error(
                  f"❌ OpenAI API Error: {str(e)}. Falling back to statistical"
                  " methods."
              )

      st.markdown("#### 📊 Statistical Anomaly Detection")
      numeric_cols = working_df.select_dtypes(include=[np.number]).columns
      if len(numeric_cols) > 0:
        selected_num_col = st.selectbox(
            "Select numeric column:", numeric_cols, key="stat_anomaly_col"
        )
        q1 = working_df[selected_num_col].quantile(0.25)
        q3 = working_df[selected_num_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = working_df[
            (working_df[selected_num_col] < lower_bound)
            | (working_df[selected_num_col] > upper_bound)
        ]

        if len(outliers) > 0:
          st.warning(f"Found {len(outliers)} outliers in {selected_num_col}")
          st.dataframe(outliers[[selected_num_col]], use_container_width=True)

          treatment = st.selectbox(
              "Outlier Treatment:",
              ["Keep", "Remove", "Cap", "Impute with median"],
              key="outlier_treatment_select",
          )

          if st.button("Apply Treatment", use_container_width=True):
            if treatment == "Remove":
              working_df = working_df[
                  (working_df[selected_num_col] >= lower_bound)
                  & (working_df[selected_num_col] <= upper_bound)
              ]
              st.success("Outliers removed!")
            elif treatment == "Cap":
              working_df[selected_num_col] = np.where(
                  working_df[selected_num_col] < lower_bound,
                  lower_bound,
                  np.where(
                      working_df[selected_num_col] > upper_bound,
                      upper_bound,
                      working_df[selected_num_col],
                  ),
              )
              st.success("Outliers capped!")
            elif treatment == "Impute with median":
              median_val = working_df[selected_num_col].median()
              working_df.loc[
                  (working_df[selected_num_col] < lower_bound)
                  | (working_df[selected_num_col] > upper_bound),
                  selected_num_col,
              ] = median_val
              st.success("Outliers imputed with median!")
        else:
          st.success(f"✅ No statistical outliers found in {selected_num_col}")

      st.markdown("#### 🛠️ Interactive Cleaning Tools")
      col1, col2 = st.columns(2)
      with col1:
        if st.button("🧹 Remove All Duplicates", use_container_width=True):
          before = len(working_df)
          working_df.drop_duplicates(inplace=True)
          after = len(working_df)
          st.success(f"Removed {before - after} duplicate rows")

      with col2:
        text_cols = working_df.select_dtypes(include=["object"]).columns.tolist()
        text_col = st.selectbox(
            "Select text column to clean:",
            [None] + text_cols,
            key="text_clean_col",
        )
        if text_col and st.button("📝 Clean Text", use_container_width=True):
          working_df[text_col] = (
              working_df[text_col].astype(str).str.strip().str.title()
          )
          st.success(f"Cleaned {text_col}: Trimmed and title-cased")

    with tab4:
      st.markdown(
          '<div class="section-header"><h3 style="margin: 0;">🎯 Smart'
          " Imputation & Transformation</h3></div>",
          unsafe_allow_html=True,
      )

      if include_smart_imputation:
        st.markdown("#### 🎯 Smart Imputation Suggestions")
        missing_cols = working_df.columns[working_df.isnull().any()].tolist()

        if missing_cols:
          st.info(f"Columns with missing values: {', '.join(missing_cols)}")
          for col in missing_cols:
            with st.expander(f"Impute {col}", expanded=True):
              col_data = working_df[col]
              missing_count = col_data.isnull().sum()
              total_count = len(col_data)

              st.caption(
                  f"Missing: {missing_count}/{total_count}"
                  f" ({missing_count/total_count*100:.1f}%)"
              )

              if pd.api.types.is_numeric_dtype(col_data):
                suggestions = [
                    f"Mean: {col_data.mean():.2f}",
                    f"Median: {col_data.median():.2f}",
                    "Forward fill",
                    "Backward fill",
                    "KNN imputation",
                ]
              else:
                suggestions = [
                    (
                        f"Mode:"
                        f" {col_data.mode()[0] if not col_data.mode().empty else 'N/A'}"
                    ),
                    "Forward fill",
                    "Backward fill",
                    "Custom value",
                ]

              selected_method = st.selectbox(
                  f"Imputation method for {col}:",
                  suggestions,
                  key=f"impute_{col}",
              )

              # Input sanitization for custom imputation values
              custom_val_input = ""
              if "Custom value" in selected_method:
                custom_val_input = st.text_input(
                    f"Enter custom value for {col}:",
                    key=f"custom_val_{col}",
                )

              if st.button(
                  f"Apply to {col}", key=f"apply_{col}", use_container_width=True
              ):
                progress_bar = st.progress(
                    0, text=f"Applying imputation to {col}..."
                )
                if "Mean" in selected_method:
                  impute_val = col_data.mean()
                  working_df[col].fillna(impute_val, inplace=True)
                elif "Median" in selected_method:
                  impute_val = col_data.median()
                  working_df[col].fillna(impute_val, inplace=True)
                elif "Mode" in selected_method:
                  impute_val = (
                      col_data.mode()[0]
                      if not col_data.mode().empty
                      else "Unknown"
                  )
                  working_df[col].fillna(impute_val, inplace=True)
                elif "Forward" in selected_method:
                  working_df[col] = col_data.ffill()
                elif "Backward" in selected_method:
                  working_df[col] = col_data.bfill()
                elif "KNN" in selected_method:
                  numeric_data = working_df.select_dtypes(
                      include=[np.number]
                  )
                  if len(numeric_data.columns) > 1:
                    imputer = KNNImputer(n_neighbors=5)
                    imputed = imputer.fit_transform(numeric_data)
                    working_df[numeric_data.columns] = imputed
                  else:
                    st.warning("Need at least 2 numeric columns for KNN")
                elif "Custom value" in selected_method:
                  if custom_val_input.strip() != "":
                    # Sanitize input if numeric required
                    if pd.api.types.is_numeric_dtype(col_data):
                      try:
                        sanitized_val = float(custom_val_input)
                        working_df[col].fillna(sanitized_val, inplace=True)
                      except ValueError:
                        st.error(
                            "❌ Invalid numeric value provided for sanitization."
                        )
                    else:
                      working_df[col].fillna(custom_val_input, inplace=True)
                  else:
                    st.error("❌ Custom value cannot be empty.")
                progress_bar.progress(100, text="Imputation complete!")
                progress_bar.empty()
                st.success(f"Successfully processed imputation for {col}!")
        else:
          st.success("✅ No missing values found!")

      st.markdown("#### 🔄 Data Transformations")
      col1, col2 = st.columns(2)
      with col1:
        convert_col = st.selectbox(
            "Select column to convert:",
            working_df.columns,
            key="convert_col_select",
        )
        new_type = st.selectbox(
            "Convert to:",
            ["int", "float", "str", "datetime", "category"],
            key="new_type_select",
        )
        if st.button("Convert Type", use_container_width=True):
          try:
            if new_type == "datetime":
              working_df[convert_col] = pd.to_datetime(
                  working_df[convert_col], errors="coerce"
              )
            elif new_type == "int":
              working_df[convert_col] = pd.to_numeric(
                  working_df[convert_col], errors="coerce"
              ).astype("Int64")
            elif new_type == "float":
              working_df[convert_col] = pd.to_numeric(
                  working_df[convert_col], errors="coerce"
              )
            elif new_type == "category":
              working_df[convert_col] = working_df[convert_col].astype(
                  "category"
              )
            else:
              working_df[convert_col] = working_df[convert_col].astype(str)
            st.success(f"Converted {convert_col} to {new_type}")
          except Exception as e:
            st.error(f"❌ Conversion failed: {str(e)}")

      with col2:
        num_cols = working_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
          norm_col = st.selectbox(
              "Select numeric column to normalize:",
              num_cols,
              key="norm_col_select",
          )
          norm_method = st.selectbox(
              "Normalization method:",
              ["Min-Max Scaling", "Z-Score", "Log Transform"],
              key="norm_method_select",
          )
          if st.button("Apply Normalization", use_container_width=True):
            if norm_method == "Min-Max Scaling":
              min_val = working_df[norm_col].min()
              max_val = working_df[norm_col].max()
              if max_val > min_val:
                working_df[f"{norm_col}_normalized"] = (
                    working_df[norm_col] - min_val
                ) / (max_val - min_val)
                st.success(f"Min-Max scaling applied to {norm_col}")
            elif norm_method == "Z-Score":
              mean_val = working_df[norm_col].mean()
              std_val = working_df[norm_col].std()
              if std_val > 0:
                working_df[f"{norm_col}_zscore"] = (
                    working_df[norm_col] - mean_val
                ) / std_val
                st.success(f"Z-Score normalization applied to {norm_col}")
            elif norm_method == "Log Transform":
              if working_df[norm_col].min() > 0:
                working_df[f"{norm_col}_log"] = np.log(working_df[norm_col])
                st.success(f"Log transform applied to {norm_col}")
              else:
                st.error("❌ Log transform requires strictly positive values.")

    with tab5:
      st.markdown(
          '<div class="section-header"><h3 style="margin: 0;">🔒'
          " Privacy-Preserving Transformations</h3></div>",
          unsafe_allow_html=True,
      )

      if include_privacy_transform:
        pii_columns = {}
        for col in working_df.columns:
          col_lower = str(col).lower()
          for pii_type, patterns in CONFIG["pii_patterns"].items():
            if any(pattern in col_lower for pattern in patterns):
              if pii_type not in pii_columns:
                pii_columns[pii_type] = []
              pii_columns[pii_type].append(col)

        if pii_columns:
          st.warning("⚠️ Potential PII Columns Detected:")
          for pii_type, cols in pii_columns.items():
            st.caption(f"**{pii_type.upper()}:** {', '.join(cols)}")

          st.markdown("#### 🛡️ Privacy Protection Options")
          selected_pii_type = st.selectbox(
              "Select PII type to protect:",
              list(pii_columns.keys()),
              key="pii_type_select",
          )
          selected_col = st.selectbox(
              "Select column:",
              pii_columns[selected_pii_type],
              key="pii_col_select",
          )

          protection_method = st.selectbox(
              "Protection method:",
              [
                  "Pseudonymization",
                  "Masking",
                  "Tokenization",
                  "Synthetic Generation",
                  "Aggregation",
              ],
              key="protection_method_select",
          )

          if st.button("🔒 Apply Protection", use_container_width=True):
            if protection_method == "Pseudonymization":
              if selected_pii_type == "email":
                fake_data = [fake.email() for _ in range(len(working_df))]
              elif selected_pii_type == "phone":
                fake_data = [
                    fake.phone_number() for _ in range(len(working_df))
                ]
              elif selected_pii_type == "name":
                fake_data = [fake.name() for _ in range(len(working_df))]
              elif selected_pii_type == "address":
                fake_data = [fake.address() for _ in range(len(working_df))]
              else:
                fake_data = [fake.uuid4() for _ in range(len(working_df))]

              working_df[f"{selected_col}_pseudonymized"] = fake_data
              st.success(f"Pseudonymized {selected_col}")

            elif protection_method == "Masking":
              if selected_pii_type == "email":
                working_df[f"{selected_col}_masked"] = working_df[
                    selected_col
                ].apply(
                    lambda x: (
                        re.sub(r"(@.*)", "@***.com", str(x))
                        if "@" in str(x)
                        else x
                    )
                )
              elif selected_pii_type in ["phone", "ssn", "credit_card"]:
                working_df[f"{selected_col}_masked"] = working_df[
                    selected_col
                ].apply(
                    lambda x: (
                        str(x)[:3]
                        + "*" * (len(str(x)) - 6)
                        + str(x)[-3:]
                        if len(str(x)) > 6
                        else x
                    )
                )
              st.success(f"Masked {selected_col}")

            elif protection_method == "Synthetic Generation":
              st.info("Generating synthetic dataset...")
              synthetic_data = working_df.copy()
              for col in working_df.columns:
                if pd.api.types.is_numeric_dtype(working_df[col]):
                  mean = working_df[col].mean()
                  std = working_df[col].std()
                  synthetic_data[col] = np.random.normal(
                      mean, std if not np.isnan(std) else 1, len(working_df)
                  )
                elif working_df[col].nunique() < 20:
                  unique_vals = working_df[col].dropna().unique()
                  if len(unique_vals) > 0:
                    synthetic_data[col] = np.random.choice(
                        unique_vals, len(working_df)
                    )
              st.session_state.synthetic_data = synthetic_data
              st.success("Synthetic dataset generated!")
              st.dataframe(synthetic_data.head(), use_container_width=True)
        else:
          st.success("✅ No obvious PII columns detected")

      if include_quality_report:
        st.markdown("#### 📊 Auto-Quality Report Generation")
        if st.button("📈 Generate Quality Report", use_container_width=True):
          with st.spinner("Generating comprehensive quality report..."):
            report = {
                "timestamp": datetime.now().isoformat(),
                "dataset_info": {
                    "rows": int(working_df.shape[0]),
                    "columns": int(working_df.shape[1]),
                    "memory_usage": float(
                        working_df.memory_usage(deep=True).sum()
                        / 1024
                        / 1024
                    ),
                },
                "quality_metrics": {
                    "completeness": float(
                        (
                            1
                            - working_df.isnull().sum().sum()
                            / (working_df.shape[0] * working_df.shape[1])
                        )
                        * 100
                    ),
                    "uniqueness": float(
                        (
                            working_df.nunique().sum()
                            / (working_df.shape[0] * working_df.shape[1])
                        )
                        * 100
                    ),
                    "duplicate_rows": int(working_df.duplicated().sum()),
                    "mixed_types": int(
                        sum([
                            len(working_df[col].apply(type).value_counts()) > 1
                            for col in working_df.columns
                        ])
                    ),
                },
                "column_analysis": {},
                "recommendations": [],
            }

            for col in working_df.columns:
              col_data = working_df[col]
              report["column_analysis"][col] = {
                  "data_type": str(col_data.dtype),
                  "missing_values": int(col_data.isnull().sum()),
                  "missing_percentage": float(
                      (col_data.isnull().sum() / len(working_df)) * 100
                  ),
                  "unique_values": int(col_data.nunique()),
                  "sample_values": col_data.dropna()
                  .head(3)
                  .astype(str)
                  .tolist(),
              }

            if report["quality_metrics"]["duplicate_rows"] > 0:
              report["recommendations"].append(
                  "Remove duplicate rows to improve data quality"
              )
            if report["quality_metrics"]["mixed_types"] > 0:
              report["recommendations"].append(
                  "Fix mixed data types in columns"
              )
            if report["quality_metrics"]["completeness"] < 90:
              report["recommendations"].append(
                  "Consider imputation for missing values"
              )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
              st.metric(
                  "Completeness",
                  f"{report['quality_metrics']['completeness']:.1f}%",
              )
            with col_b:
              st.metric(
                  "Uniqueness",
                  f"{report['quality_metrics']['uniqueness']:.1f}%",
              )
            with col_c:
              st.metric(
                  "Duplicate Rows",
                  report["quality_metrics"]["duplicate_rows"],
              )

            report_json = json.dumps(
                report, indent=2, cls=NumpyEncoder, ensure_ascii=False
            )
            st.download_button(
                label="📥 Download Quality Report (JSON)",
                data=report_json.encode("utf-8"),
                file_name="data_quality_report.json",
                mime="application/json",
                use_container_width=True,
            )

    with tab6:
      st.markdown(
          '<div class="section-header"><h3 style="margin: 0;">📤 Export'
          " Results</h3></div>",
          unsafe_allow_html=True,
      )

      st.markdown("#### 📝 Cleaned Data Preview")
      st.dataframe(working_df.head(10), use_container_width=True)

      st.markdown("#### 📤 Export Options")
      col1, col2, col3, col4 = st.columns(4)

      with col1:
        csv_data = working_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

      with col2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
          working_df.to_excel(writer, index=False, sheet_name="Cleaned Data")
          if st.session_state.original_data is not None:
            st.session_state.original_data.to_excel(
                writer, index=False, sheet_name="Original Data"
            )
        st.download_button(
            label="📊 Download Excel",
            data=output.getvalue(),
            file_name="cleaned_data.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

      with col3:
        if st.session_state.synthetic_data is not None:
          synthetic_csv = st.session_state.synthetic_data.to_csv(
              index=False
          ).encode("utf-8")
          st.download_button(
              label="🔒 Download Synthetic Data",
              data=synthetic_csv,
              file_name="synthetic_data.csv",
              mime="text/csv",
              use_container_width=True,
          )
        else:
          st.info("Generate synthetic data in Privacy tab")

      with col4:
        cleaning_log = {
            "cleaning_timestamp": datetime.now().isoformat(),
            "original_shape": {
                "rows": (
                    int(st.session_state.original_data.shape[0])
                    if st.session_state.original_data is not None
                    else None
                ),
                "columns": (
                    int(st.session_state.original_data.shape[1])
                    if st.session_state.original_data is not None
                    else None
                ),
            },
            "cleaned_shape": {
                "rows": int(working_df.shape[0]),
                "columns": int(working_df.shape[1]),
            },
            "rows_removed": (
                int(
                    st.session_state.original_data.shape[0]
                    - working_df.shape[0]
                )
                if st.session_state.original_data is not None
                else 0
            ),
            "columns_changed": list(working_df.columns),
            "missing_values_before": (
                int(st.session_state.original_data.isnull().sum().sum())
                if st.session_state.original_data is not None
                else 0
            ),
            "missing_values_after": int(working_df.isnull().sum().sum()),
            "data_types_summary": {
                col: str(working_df[col].dtype) for col in working_df.columns
            },
        }
        log_json = json.dumps(cleaning_log, indent=2, cls=NumpyEncoder)
        st.download_button(
            label="📋 Download Cleaning Log",
            data=log_json.encode("utf-8"),
            file_name="cleaning_log.json",
            mime="application/json",
            use_container_width=True,
        )

      st.markdown("#### 📊 Before & After Comparison")
      if st.session_state.original_data is not None:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
          before_missing = (
              st.session_state.original_data.isnull().sum().sum()
          )
          after_missing = working_df.isnull().sum().sum()
          st.metric(
              "Missing Values",
              f"{after_missing}",
              f"{-before_missing + after_missing}",
          )
        with col_b:
          before_dupes = st.session_state.original_data.duplicated().sum()
          after_dupes = working_df.duplicated().sum()
          st.metric(
              "Duplicate Rows",
              f"{after_dupes}",
              f"{-before_dupes + after_dupes}",
          )
        with col_c:
          before_cols = len(st.session_state.original_data.columns)
          after_cols = len(working_df.columns)
          st.metric(
              "Total Columns", f"{after_cols}", f"{after_cols - before_cols}"
          )

  st.session_state.cleaned_data = working_df

else:
  if df is not None:
    st.markdown("### 📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
      st.metric("Total Rows", df.shape[0])
      st.metric("Total Columns", df.shape[1])
    with col2:
      st.metric("Missing Values", df.isnull().sum().sum())
      st.metric(
          "Memory Usage",
          f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
      )

    st.info("Click 'Start Analysis' in the sidebar to begin processing.")
  else:
    st.info("👈 Select a data source to begin analysis.")

# Footer
st.markdown(
    """
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
        © 2026 Advanced Data Cleaning Assistant Pro
    </p>
</div>
""",
    unsafe_allow_html=True,
)