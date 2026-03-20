from datetime import datetime
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.datasets import load_iris, fetch_california_housing, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, mean_squared_error, classification_report, 
                           confusion_matrix, precision_score, recall_score, f1_score, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
import shap
import graphviz
from sklearn.tree import export_graphviz
from io import BytesIO
import json
import warnings
warnings.filterwarnings('ignore')

# Set up Streamlit App
st.set_page_config(page_title="🌳 Advanced Decision Tree Explorer", layout="wide")

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

# Header
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
    
    # Advanced settings
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

# Initialize session state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Load datasets function
def load_dataset(choice):
    """Load different datasets with preprocessing"""
    if choice == "Iris (Classification)":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        # For display only, not for modeling
        df['target_name'] = data.target_names[data.target]
        return df, 'target', 'classification', data.feature_names, data.target_names
    
    elif choice == "Wine (Classification)":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = data.target_names[data.target]
        return df, 'target', 'classification', data.feature_names, data.target_names
    
    elif choice == "Breast Cancer (Classification)":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        # Don't add target_name if it causes issues
        return df, 'target', 'classification', data.feature_names, ['Benign', 'Malignant']
    
    elif choice == "California Housing (Regression)":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'regression', data.feature_names, None
    
    elif choice == "Titanic (Classification)":
        try:
            titanic = sns.load_dataset('titanic')
            df = titanic.copy()
            
            # Preprocess
            df = df.drop(['deck', 'embark_town'], axis=1, errors='ignore')
            df = df.dropna()
            
            # Encode categorical variables
            le = LabelEncoder()
            categorical_cols = ['sex', 'embarked', 'class', 'who', 'alone', 'adult_male', 'alive']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))
            
            # Target column - make sure it's numeric
            if 'survived' in df.columns:
                df['target'] = df['survived'].astype(int)
                df = df.drop('survived', axis=1)
            
            # Remove any non-numeric columns from features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in numeric_cols:
                numeric_cols.remove('target')
            
            feature_names = numeric_cols
            
            return df, 'target', 'classification', feature_names, ['Died', 'Survived']
        except Exception as e:
            st.error(f"Error loading Titanic dataset: {e}")
            # Return Iris as fallback
            return load_dataset("Iris (Classification)")
    
    elif choice == "Diabetes (Regression)":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'regression', data.feature_names, None
    
    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            # Show column selector
            st.markdown("### 🎯 Select Target Column")
            target_column = st.selectbox("Target Column:", df.columns)
            
            # Prepare features - convert non-numeric columns
            feature_names = []
            for col in df.columns:
                if col != target_column:
                    # Try to convert to numeric
                    try:
                        df[col] = pd.to_numeric(df[col])
                        feature_names.append(col)
                    except:
                        # If conversion fails, drop the column or encode it
                        st.warning(f"Column '{col}' has non-numeric values. It will be dropped from features.")
            
            # Auto-detect task type
            unique_values = df[target_column].nunique()
            if unique_values < 10:  # Classification
                task_type = 'classification'
                # Ensure target is numeric
                le = LabelEncoder()
                df[target_column] = le.fit_transform(df[target_column].astype(str))
                target_names = le.classes_
            else:  # Regression
                task_type = 'regression'
                # Ensure target is numeric
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                target_names = None
            
            # Drop rows with NaN in target
            df = df.dropna(subset=[target_column])
            
            return df, target_column, task_type, feature_names, target_names
        
        return None, None, None, None, None

# SHAP Analysis Function
def perform_shap_analysis(model, X_train, X_test, task_type):
    """Perform SHAP analysis for model interpretability"""
    try:
        # Create SHAP explainer
        if task_type == 'classification':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            
            # Force plot for first prediction
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], 
                           matplotlib=True, show=False)
            plt.tight_layout()
            
            return fig, fig2, explainer, shap_values
            
        else:  # Regression
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            
            # Force plot for first prediction
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], 
                           matplotlib=True, show=False)
            plt.tight_layout()
            
            return fig, fig2, explainer, shap_values
            
    except Exception as e:
        st.warning(f"SHAP analysis limited: {str(e)}")
        return None, None, None, None

# Counterfactual Explanations
def generate_counterfactuals(model, X_test, y_test, task_type, feature_names, idx=0):
    """Generate counterfactual explanations for predictions"""
    try:
        if task_type == 'classification':
            # Get original prediction
            original_pred = model.predict(X_test.iloc[[idx]])[0]
            original_proba = model.predict_proba(X_test.iloc[[idx]])[0]
            
            # Find closest instance with different prediction
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
            
            # Sort by distance
            if counterfactuals:
                counterfactuals.sort(key=lambda x: x['distance'])
                best_cf = counterfactuals[0]
                
                # Find feature changes
                changes = []
                original_features = X_test.iloc[idx]
                cf_features = best_cf['features']
                
                for feature in feature_names:
                    orig_val = original_features[feature]
                    cf_val = cf_features[feature]
                    if abs(orig_val - cf_val) > 0.01 * (abs(orig_val) + 1e-10):
                        changes.append({
                            'feature': feature,
                            'original': orig_val,
                            'counterfactual': cf_val,
                            'change': cf_val - orig_val,
                            'change_pct': ((cf_val - orig_val) / (abs(orig_val) + 1e-10)) * 100
                        })
                
                return {
                    'original': {
                        'index': idx,
                        'prediction': original_pred,
                        'probability': max(original_proba),
                        'features': original_features
                    },
                    'counterfactual': best_cf,
                    'changes': changes[:5]  # Top 5 most significant changes
                }
        
        else:  # Regression
            original_pred = model.predict(X_test.iloc[[idx]])[0]
            original_value = y_test.iloc[idx]
            
            # Find similar instance with different prediction
            predictions = model.predict(X_test)
            residuals = np.abs(predictions - y_test.values)
            
            # Find instance with similar features but different residual
            distances = []
            for i in range(len(X_test)):
                if i != idx:
                    distance = np.linalg.norm(X_test.iloc[idx].values - X_test.iloc[i].values)
                    residual_diff = abs(residuals[i] - residuals[idx])
                    distances.append((i, distance, residual_diff, predictions[i]))
            
            if distances:
                distances.sort(key=lambda x: x[1])  # Sort by feature similarity
                best_idx = distances[0][0]
                best_pred = distances[0][3]
                
                # Find feature changes
                changes = []
                original_features = X_test.iloc[idx]
                cf_features = X_test.iloc[best_idx]
                
                for feature in feature_names:
                    orig_val = original_features[feature]
                    cf_val = cf_features[feature]
                    if abs(orig_val - cf_val) > 0.01 * (abs(orig_val) + 1e-10):
                        changes.append({
                            'feature': feature,
                            'original': orig_val,
                            'counterfactual': cf_val,
                            'change': cf_val - orig_val
                        })
                
                return {
                    'original': {
                        'index': idx,
                        'prediction': original_pred,
                        'actual': original_value,
                        'features': original_features
                    },
                    'counterfactual': {
                        'index': best_idx,
                        'prediction': best_pred,
                        'actual': y_test.iloc[best_idx],
                        'features': cf_features
                    },
                    'changes': changes[:5]
                }
    
    except Exception as e:
        st.warning(f"Counterfactual generation limited: {str(e)}")
        return None

# Tree Pruning Suggestions
def suggest_pruning(model, X_val, y_val, task_type):
    """Suggest optimal pruning parameters"""
    suggestions = []
    
    if task_type == 'classification':
        # Test different max_depth values
        depths = range(1, 21)
        scores = []
        
        for depth in depths:
            temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            scores.append(cross_val_score(temp_model, X_val, y_val, cv=5).mean())
        
        optimal_depth = depths[np.argmax(scores)]
        current_depth = model.get_depth()
        
        if current_depth > optimal_depth:
            suggestions.append({
                'parameter': 'max_depth',
                'current': current_depth,
                'suggested': optimal_depth,
                'reason': f'Reducing depth from {current_depth} to {optimal_depth} may improve generalization',
                'improvement': f'{max(scores)*100:.1f}% vs current {scores[current_depth-1]*100:.1f}% CV score'
            })
    
    # Suggest min_samples_leaf
    current_leaf = model.get_params().get('min_samples_leaf', 1)
    if current_leaf < 5:
        suggestions.append({
            'parameter': 'min_samples_leaf',
            'current': current_leaf,
            'suggested': min(5, len(X_val) // 20),
            'reason': 'Increasing min_samples_leaf reduces overfitting',
            'improvement': 'Better generalization on unseen data'
        })
    
    return suggestions

# Business Rule Extraction
def extract_business_rules(model, feature_names, max_rules=10):
    """Extract business rules from decision tree"""
    try:
        rules = []
        tree_rules = export_text(model, feature_names=feature_names)
        
        # Parse rules
        lines = tree_rules.split('\n')
        for line in lines:
            if 'class:' in line and '---' in line:
                # Extract rule path
                rule_path = line.split('class:')[0].strip()
                class_label = line.split('class:')[1].strip()
                
                # Clean up rule
                rule_path = rule_path.replace('|--- ', '').replace('|   ', ' AND ')
                rule_path = rule_path.replace('<=', ' ≤ ').replace('>', ' > ')
                
                # Count samples
                sample_match = re.search(r'\[(\d+)\]', line)
                samples = sample_match.group(1) if sample_match else 'N/A'
                
                if rule_path and 'AND' in rule_path:
                    rules.append({
                        'rule': f"IF {rule_path} THEN class = {class_label}",
                        'samples': samples,
                        'class': class_label
                    })
        
        return rules[:max_rules]
    except:
        return []

# Model Distillation
def distill_model(complex_model, X_train, y_train, task_type, feature_names):
    """Distill complex model into simpler one"""
    try:
        if task_type == 'classification':
            # Train a simple logistic regression on complex model's predictions
            simple_model = LogisticRegression(max_iter=1000)
            
            # Get probabilities from complex model
            y_proba = complex_model.predict_proba(X_train)
            
            # Train simple model
            simple_model.fit(y_proba, y_train)
            
            # Compare performance
            simple_acc = simple_model.score(y_proba, y_train)
            complex_acc = complex_model.score(X_train, y_train)
            
            # Extract simplified rules
            rules = []
            for i, coef in enumerate(simple_model.coef_[0]):
                if abs(coef) > 0.1:
                    rules.append({
                        'feature': feature_names[i] if i < len(feature_names) else f'Prob_Class_{i}',
                        'coefficient': coef,
                        'importance': abs(coef)
                    })
            
            rules.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'simple_model': simple_model,
                'complex_accuracy': complex_acc,
                'simple_accuracy': simple_acc,
                'rules': rules[:10],
                'accuracy_loss': (complex_acc - simple_acc) * 100
            }
        
        else:  # Regression
            # Train simple linear regression
            simple_model = LinearRegression()
            
            # Get predictions from complex model
            y_pred = complex_model.predict(X_train)
            
            # Train simple model on residuals or features
            simple_model.fit(X_train, y_pred)
            
            # Compare performance
            simple_mse = mean_squared_error(y_pred, simple_model.predict(X_train))
            complex_mse = mean_squared_error(y_train, y_pred)
            
            # Extract simplified rules
            rules = []
            for i, coef in enumerate(simple_model.coef_):
                if abs(coef) > 0.01:
                    rules.append({
                        'feature': feature_names[i],
                        'coefficient': coef,
                        'importance': abs(coef)
                    })
            
            rules.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'simple_model': simple_model,
                'complex_mse': complex_mse,
                'simple_mse': simple_mse,
                'rules': rules[:10],
                'mse_increase': (simple_mse - complex_mse)
            }
    
    except Exception as e:
        st.warning(f"Model distillation limited: {str(e)}")
        return None

# Main Analysis Execution
if dataset_choice:
    df, target_column, task_type, feature_names, target_names = load_dataset(dataset_choice)
    
    if df is not None and st.session_state.run_analysis:
        with st.spinner("🌳 Training models and analyzing..."):
            # Ensure we only use numeric columns for features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target from numeric columns if it's there
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            
            # Update feature_names to only include numeric columns
            feature_names = [col for col in feature_names if col in numeric_cols]
            
            # Use only numeric columns for X
            X = df[feature_names]
            y = df[target_column]
            
            # Data Preparation
            st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Dataset Overview</h3></div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len(feature_names))
            with col3:
                if task_type == 'classification':
                    unique_classes = df[target_column].nunique()
                    st.metric("Classes", unique_classes)
                else:
                    st.metric("Task Type", "Regression")
            with col4:
                if task_type == 'classification':
                    class_dist = df[target_column].value_counts()
                    st.metric("Majority Class", f"{class_dist.max()/len(df)*100:.1f}%")
                else:
                    st.metric("Target Range", f"{df[target_column].min():.1f}-{df[target_column].max():.1f}")
            
            # Split data
            X = df[feature_names]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Model Parameters
            st.sidebar.markdown("---")
            st.sidebar.markdown("#### 🌳 Tree Parameters")
            
            max_depth = st.sidebar.slider("Max Depth", 1, 30, 5, help="Maximum depth of the tree")
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, help="Minimum samples required to split a node")
            min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1, help="Minimum samples required at a leaf node")
            
            if task_type == 'classification':
                criterion = st.sidebar.selectbox("Criterion", ['gini', 'entropy'])
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=42
                )
            else:
                criterion = st.sidebar.selectbox("Criterion", ['squared_error', 'friedman_mse', 'absolute_error'])
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=42
                )
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Performance", 
                "🔍 SHAP Analysis", 
                "🔄 Counterfactuals",
                "✂️ Pruning & Rules",
                "🎓 Distillation",
                "📊 Visualizations"
            ])
            
            with tab1:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">Model Performance Metrics</h4></div>', unsafe_allow_html=True)
                
                if task_type == 'classification':
                    # Classification metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.3f}")
                    col2.metric("Precision", f"{precision:.3f}")
                    col3.metric("Recall", f"{recall:.3f}")
                    col4.metric("F1-Score", f"{f1:.3f}")
                    
                    # Confusion Matrix
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cm = confusion_matrix(y_test, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=target_names if target_names is not None else range(len(np.unique(y))),
                                    yticklabels=target_names if target_names is not None else range(len(np.unique(y))),
                                    ax=ax)
                        plt.title('Confusion Matrix')
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("#### 📋 Classification Report")
                        report = classification_report(y_test, y_pred, target_names=target_names if target_names is not None else None)
                        st.text(report)
                
                else:
                    # Regression metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_test - y_pred))
                    r2 = r2_score(y_test, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MSE", f"{mse:.3f}")
                    col2.metric("RMSE", f"{rmse:.3f}")
                    col3.metric("MAE", f"{mae:.3f}")
                    col4.metric("R² Score", f"{r2:.3f}")
                    
                    # Residual plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Residuals histogram
                    residuals = y_test - y_pred
                    ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
                    ax1.set_xlabel('Residuals')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('Residual Distribution')
                    
                    # Actual vs Predicted
                    ax2.scatter(y_test, y_pred, alpha=0.6)
                    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax2.set_xlabel('Actual')
                    ax2.set_ylabel('Predicted')
                    ax2.set_title('Actual vs Predicted')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Feature Importance
                st.markdown("#### 🎯 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
                ax.set_title('Top 10 Feature Importances')
                st.pyplot(fig)
            
            with tab2:
                if include_shap:
                    st.markdown('<div class="section-header"><h4 style="margin: 0;">🔍 SHAP Value Analysis</h4></div>', unsafe_allow_html=True)
                    
                    # Perform SHAP analysis
                    shap_fig, force_fig, explainer, shap_values = perform_shap_analysis(model, X_train, X_test, task_type)
                    
                    if shap_fig:
                        st.markdown("#### 📊 SHAP Summary Plot")
                        st.pyplot(shap_fig)
                        
                        st.markdown("#### 💪 SHAP Force Plot (First Sample)")
                        st.pyplot(force_fig)
                        
                        # SHAP Dependence plots
                        if len(feature_names) >= 2:
                            st.markdown("#### 🔗 SHAP Dependence Plots")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_dep1, ax = plt.subplots(figsize=(8, 6))
                                shap.dependence_plot(0, shap_values if task_type == 'regression' else shap_values[0], 
                                                    X_test, show=False)
                                plt.tight_layout()
                                st.pyplot(fig_dep1)
                            
                            with col2:
                                fig_dep2, ax = plt.subplots(figsize=(8, 6))
                                shap.dependence_plot(1, shap_values if task_type == 'regression' else shap_values[0], 
                                                    X_test, show=False)
                                plt.tight_layout()
                                st.pyplot(fig_dep2)
                        
                        # SHAP Waterfall plot
                        st.markdown("#### 🌊 SHAP Waterfall Plot")
                        if task_type == 'regression':
                            fig_waterfall, ax = plt.subplots(figsize=(10, 6))
                            shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                               base_values=explainer.expected_value, 
                                                               data=X_test.iloc[0]), show=False)
                            plt.tight_layout()
                            st.pyplot(fig_waterfall)
                
                else:
                    st.info("Enable SHAP Analysis in sidebar to see this section.")
            
            with tab3:
                if include_counterfactuals:
                    st.markdown('<div class="section-header"><h4 style="margin: 0;">🔄 Counterfactual Explanations</h4></div>', unsafe_allow_html=True)
                    
                    # Generate counterfactuals
                    cf_result = generate_counterfactuals(model, X_test, y_test, task_type, feature_names)
                    
                    if cf_result:
                        st.markdown("#### 📝 Original Prediction")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Original Instance:**")
                            st.write(f"Index: {cf_result['original']['index']}")
                            
                            if task_type == 'classification':
                                st.write(f"Predicted Class: {cf_result['original']['prediction']}")
                                if target_names is not None:
                                    st.write(f"Class Name: {target_names[cf_result['original']['prediction']]}")
                                st.write(f"Confidence: {cf_result['original']['probability']:.3f}")
                            else:
                                st.write(f"Predicted Value: {cf_result['original']['prediction']:.3f}")
                                st.write(f"Actual Value: {cf_result['original']['actual']:.3f}")
                        
                        with col2:
                            st.markdown("**Counterfactual Instance:**")
                            st.write(f"Index: {cf_result['counterfactual']['index']}")
                            
                            if task_type == 'classification':
                                st.write(f"Predicted Class: {cf_result['counterfactual']['prediction']}")
                                if target_names is not None:
                                    st.write(f"Class Name: {target_names[cf_result['counterfactual']['prediction']]}")
                                st.write(f"Confidence: {cf_result['counterfactual']['probability']:.3f}")
                            else:
                                st.write(f"Predicted Value: {cf_result['counterfactual']['prediction']:.3f}")
                                st.write(f"Actual Value: {cf_result['counterfactual']['actual']:.3f}")
                        
                        st.markdown("#### 🔀 Feature Changes Needed")
                        
                        changes_df = pd.DataFrame(cf_result['changes'])
                        if not changes_df.empty:
                            st.dataframe(changes_df, use_container_width=True)
                            
                            # Visualize changes
                            if len(changes_df) > 0:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                x = np.arange(len(changes_df))
                                width = 0.35
                                
                                ax.bar(x - width/2, changes_df['original'], width, label='Original', alpha=0.7)
                                ax.bar(x + width/2, changes_df['counterfactual'], width, label='Counterfactual', alpha=0.7)
                                
                                ax.set_xlabel('Features')
                                ax.set_ylabel('Values')
                                ax.set_title('Feature Changes for Different Outcome')
                                ax.set_xticks(x)
                                ax.set_xticklabels(changes_df['feature'], rotation=45, ha='right')
                                ax.legend()
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                
                else:
                    st.info("Enable Counterfactual Explanations in sidebar to see this section.")
            
            with tab4:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">✂️ Tree Optimization & Business Rules</h4></div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if include_pruning:
                        st.markdown("#### 🌳 Tree Pruning Suggestions")
                        pruning_suggestions = suggest_pruning(model, X_train, y_train, task_type)
                        
                        if pruning_suggestions:
                            for suggestion in pruning_suggestions:
                                with st.container():
                                    st.markdown(f"**Parameter:** {suggestion['parameter']}")
                                    st.markdown(f"Current: `{suggestion['current']}` → Suggested: `{suggestion['suggested']}`")
                                    st.caption(suggestion['reason'])
                                    st.caption(f"Expected: {suggestion['improvement']}")
                                    st.divider()
                        else:
                            st.success("Tree parameters are well-tuned!")
                    
                    # Tree Complexity Analysis
                    st.markdown("#### 📊 Tree Complexity")
                    tree_depth = model.get_depth()
                    n_leaves = model.get_n_leaves()
                    n_nodes = model.tree_.node_count
                    
                    col1a, col2a, col3a = st.columns(3)
                    col1a.metric("Depth", tree_depth)
                    col2a.metric("Leaves", n_leaves)
                    col3a.metric("Nodes", n_nodes)
                    
                    # Complexity vs Performance
                    if task_type == 'classification':
                        depths = range(1, min(20, len(X_train)))
                        train_scores = []
                        test_scores = []
                        
                        for depth in depths:
                            temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
                            temp_model.fit(X_train, y_train)
                            train_scores.append(temp_model.score(X_train, y_train))
                            test_scores.append(temp_model.score(X_test, y_test))
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(depths, train_scores, label='Train', marker='o')
                        ax.plot(depths, test_scores, label='Test', marker='s')
                        ax.axvline(x=max_depth, color='r', linestyle='--', label=f'Current ({max_depth})')
                        ax.set_xlabel('Tree Depth')
                        ax.set_ylabel('Accuracy')
                        ax.set_title('Model Performance vs Tree Depth')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                
                with col2:
                    if include_business_rules:
                        st.markdown("#### 📋 Business Rule Extraction")
                        business_rules = extract_business_rules(model, feature_names)
                        
                        if business_rules:
                            for i, rule in enumerate(business_rules[:5], 1):
                                with st.container():
                                    st.markdown(f"**Rule {i}**")
                                    st.caption(rule['rule'])
                                    st.caption(f"Samples: {rule['samples']} | Class: {rule['class']}")
                                    st.divider()
                        
                        # Download rules
                        if business_rules:
                            rules_df = pd.DataFrame(business_rules)
                            csv_rules = rules_df.to_csv(index=False).encode()
                            st.download_button(
                                "📥 Download Rules",
                                csv_rules,
                                "business_rules.csv",
                                "text/csv",
                                use_container_width=True
                            )
                    
                    # Rule coverage analysis
                    st.markdown("#### 📈 Rule Coverage Analysis")
                    
                    # Get leaf node statistics
                    leaf_samples = model.tree_.n_node_samples[model.tree_.children_left == -1]
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(leaf_samples, bins=20, edgecolor='black', alpha=0.7)
                    ax.axvline(x=min_samples_leaf, color='r', linestyle='--', label=f'Min Leaf ({min_samples_leaf})')
                    ax.set_xlabel('Samples per Leaf')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Leaf Node Sample Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            with tab5:
                if include_distillation:
                    st.markdown('<div class="section-header"><h4 style="margin: 0;">🎓 Model Distillation</h4></div>', unsafe_allow_html=True)
                    
                    # Perform model distillation
                    distillation_result = distill_model(model, X_train, y_train, task_type, feature_names)
                    
                    if distillation_result:
                        st.markdown("#### 📊 Distillation Performance")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if task_type == 'classification':
                                st.metric("Complex Model Accuracy", f"{distillation_result['complex_accuracy']:.3%}")
                                st.metric("Simple Model Accuracy", f"{distillation_result['simple_accuracy']:.3%}")
                                st.metric("Accuracy Loss", f"{distillation_result['accuracy_loss']:.2f}%")
                            else:
                                st.metric("Complex Model MSE", f"{distillation_result['complex_mse']:.3f}")
                                st.metric("Simple Model MSE", f"{distillation_result['simple_mse']:.3f}")
                                st.metric("MSE Increase", f"{distillation_result['mse_increase']:.3f}")
                        
                        with col2:
                            st.markdown("**Key Insights:**")
                            if task_type == 'classification':
                                if distillation_result['accuracy_loss'] < 5:
                                    st.success("Good distillation! Minimal accuracy loss.")
                                elif distillation_result['accuracy_loss'] < 15:
                                    st.warning("Moderate accuracy loss. Consider keeping complex model.")
                                else:
                                    st.error("Significant accuracy loss. Complex model needed.")
                            else:
                                if distillation_result['mse_increase'] < 0.1:
                                    st.success("Good distillation! Minimal MSE increase.")
                                elif distillation_result['mse_increase'] < 0.5:
                                    st.warning("Moderate MSE increase.")
                                else:
                                    st.error("Significant MSE increase.")
                        
                        st.markdown("#### 📋 Simplified Rules (Top 10)")
                        
                        rules_df = pd.DataFrame(distillation_result['rules'])
                        st.dataframe(rules_df, use_container_width=True)
                        
                        # Visualize rule importance
                        if len(distillation_result['rules']) > 0:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            top_rules = distillation_result['rules'][:10]
                            features = [r['feature'] for r in top_rules]
                            importance = [r['importance'] for r in top_rules]
                            
                            y_pos = np.arange(len(features))
                            ax.barh(y_pos, importance, alpha=0.7)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(features)
                            ax.set_xlabel('Rule Importance (abs coefficient)')
                            ax.set_title('Top 10 Simplified Rules')
                            ax.invert_yaxis()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                
                else:
                    st.info("Enable Model Distillation in sidebar to see this section.")
            
            with tab6:
                st.markdown('<div class="section-header"><h4 style="margin: 0;">📊 Model Visualizations</h4></div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🌳 Decision Tree Visualization")
                    
                    # Interactive depth selection for tree plot
                    plot_depth = st.slider("Tree Plot Depth", 1, min(5, max_depth), min(3, max_depth))
                    
                    class_names = list(target_names) if target_names is not None else None
                    
                    fig, ax = plt.subplots(figsize=(15, 10))
                    plot_tree(model, 
                             feature_names=feature_names,
                             class_names=class_names,
                             filled=True,
                             rounded=True,
                             max_depth=plot_depth,
                             ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("#### 📈 Learning Curves")
                    
                    # Generate learning curve data
                    train_sizes = np.linspace(0.1, 1.0, 10)
                    train_scores = []
                    test_scores = []
                    
                    for size in train_sizes:
                        n_samples = int(size * len(X_train))
                        X_subset = X_train[:n_samples]
                        y_subset = y_train[:n_samples]
                        
                        temp_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42) if task_type == 'classification' else DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                        temp_model.fit(X_subset, y_subset)
                        
                        train_scores.append(temp_model.score(X_subset, y_subset))
                        test_scores.append(temp_model.score(X_test, y_test))
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(train_sizes * 100, train_scores, 'o-', label='Train', linewidth=2)
                    ax.plot(train_sizes * 100, test_scores, 's-', label='Test', linewidth=2)
                    ax.set_xlabel('Training Set Size (%)')
                    ax.set_ylabel('Score')
                    ax.set_title('Learning Curves')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # 3D Feature Space Visualization (if enough features)
                if len(feature_names) >= 3:
                    st.markdown("#### 🎨 3D Feature Space")
                    
                    # Select top 3 important features
                    top_features = importance_df.head(3)['Feature'].tolist()
                    
                    if len(top_features) == 3:
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # Scatter plot
                        scatter = ax.scatter(
                            X_test[top_features[0]],
                            X_test[top_features[1]],
                            X_test[top_features[2]],
                            c=y_pred if task_type == 'regression' else y_pred,
                            cmap='viridis' if task_type == 'regression' else 'Set2',
                            alpha=0.6,
                            s=50
                        )
                        
                        ax.set_xlabel(top_features[0])
                        ax.set_ylabel(top_features[1])
                        ax.set_zlabel(top_features[2])
                        ax.set_title('3D Feature Space with Predictions')
                        
                        plt.colorbar(scatter, ax=ax, label='Predicted Value' if task_type == 'regression' else 'Class')
                        st.pyplot(fig)
            
            # Export Section
            st.markdown('<div class="section-header"><h4 style="margin: 0;">📥 Export Results</h4></div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Export predictions
                output_df = X_test.copy()
                output_df['Actual'] = y_test
                output_df['Predicted'] = y_pred
                if task_type == 'classification' and target_names is not None:
                    output_df['Actual_Class'] = [target_names[i] for i in y_test]
                    output_df['Predicted_Class'] = [target_names[i] for i in y_pred]
                
                csv_data = output_df.to_csv(index=False).encode()
                st.download_button(
                    "📄 Predictions CSV",
                    csv_data,
                    "model_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export model parameters
                model_info = {
                    'model_type': 'Decision Tree',
                    'task_type': task_type,
                    'parameters': model.get_params(),
                    'performance': {
                        'accuracy' if task_type == 'classification' else 'mse': 
                        accuracy_score(y_test, y_pred) if task_type == 'classification' else mean_squared_error(y_test, y_pred)
                    },
                    'feature_importance': importance_df.to_dict('records'),
                    'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                json_data = json.dumps(model_info, indent=2)
                st.download_button(
                    "📊 Model Info JSON",
                    json_data.encode(),
                    "model_information.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Export tree visualization as DOT
                dot_data = export_graphviz(model, feature_names=feature_names, 
                                          class_names=target_names if target_names is not None else None,
                                          filled=True, rounded=True, special_characters=True)
                st.download_button(
                    "🌳 Tree DOT File",
                    dot_data.encode(),
                    "decision_tree.dot",
                    "text/plain",
                    use_container_width=True
                )
            
            with col4:
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.session_state.run_analysis = False
                    st.rerun()
    
    elif df is not None:
        # Show data preview before analysis
        st.markdown("### 📋 Data Preview")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.metric("Total Rows", len(df))
            st.metric("Columns", len(df.columns))
            if task_type:
                st.metric("Task Type", task_type.title())
        
        st.info(f"Click 'Train & Analyze' in the sidebar to begin analysis on {len(df)} samples.")

else:
    st.info("👈 Select a dataset from the sidebar to begin.")

# Footer
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">

<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
        <span>🔍 <strong>SHAP Analysis</strong></span>
        <span>🔄 <strong>Counterfactuals</strong></span>
        <span>✂️ <strong>Tree Pruning</strong></span>
        <span>📋 <strong>Business Rules</strong></span>
        <span>🎓 <strong>Model Distillation</strong></span>
    </div>
    <p style="margin: 0.5rem 0; color: #4b5563;">
        Advanced interpretable machine learning for transparent decision-making
    </p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">
        © 2024 Advanced Decision Tree Explorer Pro
    </p>
</div>
""", unsafe_allow_html=True)