from datetime import datetime
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.datasets import load_iris, fetch_california_housing, load_wine, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, mean_squared_error, classification_report, 
                           confusion_matrix, precision_score, recall_score, f1_score, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
from sklearn.tree import export_graphviz
import json
import warnings
warnings.filterwarnings('ignore')

# Set up Streamlit App configuration
st.set_page_config(page_title="🌳 Advanced Decision Tree Explorer", layout="wide")

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
        border-left: 4px solid #10b981;
    }
    .section-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🌳 Advanced Decision Tree Explorer Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Interpretable Machine Learning with SHAP, Counterfactuals & Model Distillation
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 📊 Dataset Selection")
    dataset_choice = st.selectbox(
        "Choose Dataset:",
        ["Iris (Classification)", "Wine (Classification)", "Breast Cancer (Classification)", 
         "California Housing (Regression)", "Titanic (Classification)", "Diabetes (Regression)",
         "Upload Your Own CSV"]
    )
    
    st.markdown("---")
    st.markdown("#### 🔧 Model Settings")
    model_type = st.radio(
        "Model Type:",
        ["Decision Tree", "Random Forest", "XGBoost (if available)", "Compare Multiple"]
    )
    
    st.markdown("---")
    st.markdown("#### 🎯 Advanced Features")
    include_shap = st.checkbox("🔍 SHAP Analysis", value=True)
    include_counterfactuals = st.checkbox("🔄 Counterfactual Explanations", value=True)
    include_pruning = st.checkbox("✂️ Tree Pruning Suggestions", value=True)
    include_business_rules = st.checkbox("📋 Business Rule Extraction", value=True)
    include_distillation = st.checkbox("🎓 Model Distillation", value=True)
    
    st.markdown("---")
    if st.button("🚀 Train & Analyze", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
        st.session_state.model_trained = False  # Trigger retraining

# Initialize session state flags
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Cached dataset loader for pre-built datasets
@st.cache_data
def load_built_in_dataset(choice: str):
    if choice == "Iris (Classification)":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'classification', list(data.feature_names), list(data.target_names)
    elif choice == "Wine (Classification)":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'classification', list(data.feature_names), list(data.target_names)
    elif choice == "Breast Cancer (Classification)":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'classification', list(data.feature_names), ['Benign', 'Malignant']
    elif choice == "California Housing (Regression)":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'regression', list(data.feature_names), None
    elif choice == "Diabetes (Regression)":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'regression', list(data.feature_names), None
    elif choice == "Titanic (Classification)":
        try:
            titanic = sns.load_dataset('titanic')
            df = titanic.copy().drop(['deck', 'embark_town'], axis=1, errors='ignore').dropna()
            le = LabelEncoder()
            for col in ['sex', 'embarked', 'class', 'who', 'alone', 'adult_male', 'alive']:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))
            if 'survived' in df.columns:
                df['target'] = df['survived'].astype(int)
                df = df.drop('survived', axis=1)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in numeric_cols:
                numeric_cols.remove('target')
            return df, 'target', 'classification', numeric_cols, ['Died', 'Survived']
        except Exception:
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df, 'target', 'classification', list(data.feature_names), list(data.target_names)
    return None, None, None, None, None

def load_user_dataset(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 🎯 Select Target Column")
        target_column = st.selectbox("Target Column:", df.columns)
        feature_names = []
        for col in df.columns:
            if col != target_column:
                try:
                    df[col] = pd.to_numeric(df[col])
                    feature_names.append(col)
                except Exception:
                    pass
        unique_values = df[target_column].nunique()
        if unique_values < 10:
            task_type = 'classification'
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column].astype(str))
            target_names = list(le.classes_)
        else:
            task_type = 'regression'
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            target_names = None
        df = df.dropna(subset=[target_column])
        return df, target_column, task_type, feature_names, target_names
    return None, None, None, None, None

# Helper Analyses Functions
def perform_shap_analysis(model, X_train, X_test, task_type):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        val_idx = 0 if task_type == 'classification' else shap_values[0]
        shap.force_plot(explainer.expected_value[0] if task_type == 'classification' else explainer.expected_value, 
                        shap_values[0][0,:] if task_type == 'classification' else shap_values[0,:], 
                        X_test.iloc[0,:], matplotlib=True, show=False)
        plt.tight_layout()
        return fig, fig2, explainer, shap_values
    except Exception as e:
        st.warning(f"SHAP analysis limited: {str(e)}")
        return None, None, None, None

def generate_counterfactuals(model, X_test, y_test, task_type, feature_names, idx=0):
    try:
        if task_type == 'classification':
            original_pred = model.predict(X_test.iloc[[idx]])[0]
            original_proba = model.predict_proba(X_test.iloc[[idx]])[0]
            counterfactuals = []
            for i in range(len(X_test)):
                if i != idx and model.predict(X_test.iloc[[i]])[0] != original_pred:
                    counterfactuals.append({
                        'index': i,
                        'prediction': model.predict(X_test.iloc[[i]])[0],
                        'probability': max(model.predict_proba(X_test.iloc[[i]])[0]),
                        'distance': np.linalg.norm(X_test.iloc[idx].values - X_test.iloc[i].values),
                        'features': X_test.iloc[i]
                    })
            if counterfactuals:
                counterfactuals.sort(key=lambda x: x['distance'])
                best_cf = counterfactuals[0]
                changes = []
                for feature in feature_names:
                    orig_val = X_test.iloc[idx][feature]
                    cf_val = best_cf['features'][feature]
                    if abs(orig_val - cf_val) > 0.01 * (abs(orig_val) + 1e-10):
                        changes.append({
                            'feature': feature,
                            'original': orig_val,
                            'counterfactual': cf_val,
                            'change': cf_val - orig_val
                        })
                return {
                    'original': {'index': idx, 'prediction': original_pred, 'probability': max(original_proba)},
                    'counterfactual': best_cf,
                    'changes': changes[:5]
                }
        else:
            original_pred = model.predict(X_test.iloc[[idx]])[0]
            original_value = y_test.iloc[idx]
            predictions = model.predict(X_test)
            distances = []
            for i in range(len(X_test)):
                if i != idx:
                    dist = np.linalg.norm(X_test.iloc[idx].values - X_test.iloc[i].values)
                    distances.append((i, dist, predictions[i]))
            if distances:
                distances.sort(key=lambda x: x[1])
                best_idx = distances[0][0]
                changes = []
                for feature in feature_names:
                    orig_val = X_test.iloc[idx][feature]
                    cf_val = X_test.iloc[best_idx][feature]
                    if abs(orig_val - cf_val) > 0.01 * (abs(orig_val) + 1e-10):
                        changes.append({'feature': feature, 'original': orig_val, 'counterfactual': cf_val})
                return {
                    'original': {'index': idx, 'prediction': original_pred, 'actual': original_value},
                    'counterfactual': {'index': best_idx, 'prediction': distances[0][2], 'actual': y_test.iloc[best_idx]},
                    'changes': changes[:5]
                }
    except Exception as e:
        st.warning(f"Counterfactual generation limited: {str(e)}")
        return None

def suggest_pruning(model, X_val, y_val, task_type):
    suggestions = []
    if task_type == 'classification':
        depths = range(1, 21)
        scores = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), X_val, y_val, cv=5).mean() for d in depths]
        optimal_depth = depths[np.argmax(scores)]
        current_depth = model.get_depth()
        if current_depth > optimal_depth:
            suggestions.append({
                'parameter': 'max_depth',
                'current': current_depth,
                'suggested': optimal_depth,
                'reason': f'Reducing depth from {current_depth} to {optimal_depth} improves generalization',
                'improvement': f'Higher cross-validation accuracy expected'
            })
    current_leaf = model.get_params().get('min_samples_leaf', 1)
    if current_leaf < 5:
        suggestions.append({
            'parameter': 'min_samples_leaf',
            'current': current_leaf,
            'suggested': min(5, len(X_val) // 20),
            'reason': 'Increasing min_samples_leaf reduces overfitting',
            'improvement': 'Better robustness on unseen data'
        })
    return suggestions

def extract_business_rules(model, feature_names, max_rules=10):
    try:
        rules = []
        tree_rules = export_text(model, feature_names=feature_names)
        for line in tree_rules.split('\n'):
            if 'class:' in line and '---' in line:
                rule_path = line.split('class:')[0].strip().replace('|--- ', '').replace('|   ', ' AND ')
                rule_path = rule_path.replace('<=', ' ≤ ').replace('>', ' > ')
                class_label = line.split('class:')[1].strip()
                match = re.search(r'\[(\d+)\]', line)
                samples = match.group(1) if match else 'N/A'
                if rule_path and 'AND' in rule_path:
                    rules.append({'rule': f"IF {rule_path} THEN class = {class_label}", 'samples': samples, 'class': class_label})
        return rules[:max_rules]
    except Exception:
        return []

def distill_model(complex_model, X_train, y_train, task_type, feature_names):
    try:
        if task_type == 'classification':
            simple_model = LogisticRegression(max_iter=1000)
            y_proba = complex_model.predict_proba(X_train)
            simple_model.fit(y_proba, y_train)
            simple_acc = simple_model.score(y_proba, y_train)
            complex_acc = complex_model.score(X_train, y_train)
            rules = [{'feature': feature_names[i] if i < len(feature_names) else f'Prob_{i}', 'importance': abs(coef)} 
                     for i, coef in enumerate(simple_model.coef_[0]) if abs(coef) > 0.1]
            return {'complex_accuracy': complex_acc, 'simple_accuracy': simple_acc, 'rules': sorted(rules, key=lambda x: x['importance'], reverse=True)[:10], 'accuracy_loss': (complex_acc - simple_acc) * 100}
        else:
            simple_model = LinearRegression()
            y_pred = complex_model.predict(X_train)
            simple_model.fit(X_train, y_pred)
            simple_mse = mean_squared_error(y_pred, simple_model.predict(X_train))
            complex_mse = mean_squared_error(y_train, y_pred)
            rules = [{'feature': feature_names[i], 'importance': abs(coef)} for i, coef in enumerate(simple_model.coef_) if abs(coef) > 0.01]
            return {'complex_mse': complex_mse, 'simple_mse': simple_mse, 'rules': sorted(rules, key=lambda x: x['importance'], reverse=True)[:10], 'mse_increase': (simple_mse - complex_mse)}
    except Exception as e:
        st.warning(f"Distillation limited: {str(e)}")
        return None

# Load Dataset Logic
if dataset_choice == "Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    df, target_column, task_type, feature_names, target_names = load_user_dataset(uploaded_file)
else:
    df, target_column, task_type, feature_names, target_names = load_built_in_dataset(dataset_choice)

if df is not None:
    if st.session_state.run_analysis:
        # Session state persistence check to avoid retraining across tab clicks unless settings change
        if not st.session_state.get('model_trained', False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing data features...")
            progress_bar.progress(25)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            feature_names = [col for col in feature_names if col in numeric_cols]
            
            X = df[feature_names]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            status_text.text("Configuring model parameters...")
            progress_bar.progress(50)
            
            with st.sidebar:
                st.markdown("---")
                st.markdown("#### 🌳 Tree Parameters")
                max_depth = st.slider("Max Depth", 1, 30, 5)
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
                
                if task_type == 'classification':
                    criterion = st.selectbox("Criterion", ['gini', 'entropy'])
                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=42)
                else:
                    criterion = st.selectbox("Criterion", ['squared_error', 'friedman_mse', 'absolute_error'])
                    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=42)
            
            status_text.text("Training model...")
            progress_bar.progress(80)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store in session state to persist through tab switching
            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.feature_names = feature_names
            st.session_state.task_type = task_type
            st.session_state.target_names = target_names
            st.session_state.max_depth = max_depth
            st.session_state.min_samples_leaf = min_samples_leaf
            st.session_state.model_trained = True
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

        # Retrieve from session state
        model = st.session_state.model
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        feature_names = st.session_state.feature_names
        task_type = st.session_state.task_type
        target_names = st.session_state.target_names
        max_depth = st.session_state.max_depth
        min_samples_leaf = st.session_state.min_samples_leaf

        # Overview Metrics Section
        st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Dataset Overview</h3></div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(df))
        col2.metric("Features", len(feature_names))
        if task_type == 'classification':
            col3.metric("Classes", df[target_column].nunique())
            col4.metric("Majority Class", f"{df[target_column].value_counts().max()/len(df)*100:.1f}%")
        else:
            col3.metric("Task Type", "Regression")
            col4.metric("Target Range", f"{df[target_column].min():.1f} - {df[target_column].max():.1f}")

        # Tabs Layout
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Performance", "🔍 SHAP Analysis", "🔄 Counterfactuals",
            "✂️ Pruning & Rules", "🎓 Distillation", "📊 Visualizations"
        ])
        
        with tab1:
            st.markdown('<div class="section-header"><h4 style="margin: 0;">Model Performance Metrics</h4></div>', unsafe_allow_html=True)
            if task_type == 'classification':
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Precision", f"{prec:.3f}")
                c3.metric("Recall", f"{rec:.3f}")
                c4.metric("F1-Score", f"{f1:.3f}")
                
                colA, colB = st.columns([3, 2])
                with colA:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.title('Confusion Matrix')
                    st.pyplot(fig)
                with colB:
                    st.markdown("#### 📋 Classification Report")
                    st.text(classification_report(y_test, y_pred))
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                r2 = r2_score(y_test, y_pred)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MSE", f"{mse:.3f}")
                c2.metric("RMSE", f"{rmse:.3f}")
                c3.metric("MAE", f"{mae:.3f}")
                c4.metric("R² Score", f"{r2:.3f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.hist(y_test - y_pred, bins=30, edgecolor='black', alpha=0.7)
                ax1.set_title('Residual Distribution')
                ax2.scatter(y_test, y_pred, alpha=0.6)
                ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax2.set_title('Actual vs Predicted')
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("#### 🎯 Feature Importance")
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
            st.pyplot(fig)

        with tab2:
            if include_shap:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">🔍 SHAP Value Analysis</h4></div>', unsafe_allow_html=True)
                with st.spinner("Computing SHAP values..."):
                    shap_fig, force_fig, _, _ = perform_shap_analysis(model, X_train, X_test, task_type)
                    if shap_fig:
                        st.pyplot(shap_fig)
                        st.markdown("#### 💪 SHAP Force Plot")
                        st.pyplot(force_fig)
            else:
                st.info("Enable SHAP Analysis in sidebar to view.")

        with tab3:
            if include_counterfactuals:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">🔄 Counterfactual Explanations</h4></div>', unsafe_allow_html=True)
                cf_result = generate_counterfactuals(model, X_test, y_test, task_type, feature_names)
                if cf_result:
                    col1, col2 = st.columns(2)
                    col1.write("**Original Prediction Summary:**")
                    col1.json(cf_result['original'])
                    col2.write("**Counterfactual Alternative:**")
                    col2.json(cf_result['counterfactual'])
                    st.markdown("#### 🔀 Required Feature Changes")
                    st.dataframe(pd.DataFrame(cf_result['changes']), use_container_width=True)
            else:
                st.info("Enable Counterfactual Explanations in sidebar to view.")

        with tab4:
            st.markdown('<div class="section-header"><h4 style="margin: 0;">✂️ Tree Optimization & Business Rules</h4></div>', unsafe_allow_html=True)
            colA, colB = st.columns(2)
            with colA:
                if include_pruning:
                    st.markdown("#### 🌳 Pruning Suggestions")
                    for sug in suggest_pruning(model, X_train, y_train, task_type):
                        st.markdown(f"**Parameter:** `{sug['parameter']}` | Current: `{sug['current']}` → Suggested: `{sug['suggested']}`")
                        st.caption(sug['reason'])
                        st.divider()
            with colB:
                if include_business_rules:
                    st.markdown("#### 📋 Extracted Rules")
                    rules = extract_business_rules(model, feature_names)
                    for rule in rules[:5]:
                        st.caption(rule['rule'])
                        st.divider()

        with tab5:
            if include_distillation:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">🎓 Model Distillation</h4></div>', unsafe_allow_html=True)
                dist_res = distill_model(model, X_train, y_train, task_type, feature_names)
                if dist_res:
                    st.json(dist_res)
            else:
                st.info("Enable Model Distillation in sidebar to view.")

        with tab6:
            st.markdown('<div class="section-header"><h4 style="margin: 0;">📊 Model Visualizations</h4></div>', unsafe_allow_html=True)
            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### 🌳 Decision Tree Plot")
                # Fix tree depth slider fallback edge cases safely
                max_slider_val = max(1, min(5, max_depth))
                default_slider_val = min(3, max_depth)
                plot_depth = st.slider("Tree Plot Depth", 1, max_slider_val, default_slider_val)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_tree(model, feature_names=feature_names, class_names=list(target_names) if target_names else None, filled=True, max_depth=plot_depth, ax=ax)
                st.pyplot(fig)
            with colB:
                st.markdown("#### 📈 Complexity Overview")
                st.metric("Tree Depth", model.get_depth())
                st.metric("Leaf Nodes", model.get_n_leaves())
    else:
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.info("👈 Click **'Train & Analyze'** in the sidebar to run the explorer.")
else:
    st.info("👈 Select or upload a dataset from the sidebar to begin.")

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">
<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <p style="margin: 0.5rem 0; color: #4b5563;">Advanced interpretable machine learning for transparent decision-making</p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">© 2026 Advanced Decision Tree Explorer Pro</p>
</div>
""", unsafe_allow_html=True)