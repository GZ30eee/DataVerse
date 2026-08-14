from collections import Counter
from io import StringIO
import time
import typing
import openai
import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="📊 Smart Market Analyzer", layout="wide", page_icon="🛍️")

OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")  # Loaded from secrets.toml

# Initialize OpenAI client if key is provided
client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ==========================================
# UTILITY & PROCESSING FUNCTIONS
# ==========================================

def convert_frozenset_to_string(obj: typing.Any) -> str:
    """Convert frozenset to string safely for display and export."""
    if isinstance(obj, frozenset):
        return ', '.join(sorted(list(obj)))
    return str(obj)

def safe_df_for_plotly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all frozensets to strings for Plotly JSON compatibility."""
    df_safe = df.copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == 'object':
            df_safe[col] = df_safe[col].apply(lambda x: convert_frozenset_to_string(x))
    return df_safe

@st.cache_data(show_spinner=False)
def preprocess_transactions(transactions_list: list) -> pd.DataFrame:
    """Preprocess transaction list into a one-hot encoded boolean DataFrame with caching."""
    df_encoded = pd.DataFrame(transactions_list).stack().str.get_dummies().groupby(level=0).sum().astype(bool)
    return df_encoded

@st.cache_data(show_spinner=False)
def run_association_mining(df_encoded_hash: tuple, algorithm: str, min_support: float, min_confidence: float, min_lift: float) -> pd.DataFrame:
    """Run Apriori or FP-Growth algorithm with caching based on encoded data shape/values and thresholds."""
    # Reconstruct dataframe from cached tuple signature if needed, or pass df directly (Streamlit hashes dataframes automatically)
    pass

# Using direct dataframe caching since st.cache_data handles pandas DataFrames natively
@st.cache_data(show_spinner=False)
def cached_mining(df: pd.DataFrame, algorithm: str, min_support: float, min_confidence: float, min_lift: float) -> tuple[pd.DataFrame, str]:
    """Execute frequent itemset mining and association rules generation with error handling."""
    rules = pd.DataFrame()
    algo_used = algorithm
    
    try:
        if algorithm == "Both":
            freq_ap = apriori(df, min_support=min_support, use_colnames=True)
            rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=min_confidence) if not freq_ap.empty else pd.DataFrame()
            if not rules_ap.empty:
                rules_ap = rules_ap[rules_ap['lift'] >= min_lift]

            freq_fp = fpgrowth(df, min_support=min_support, use_colnames=True)
            rules_fp = association_rules(freq_fp, metric="confidence", min_threshold=min_confidence) if not freq_fp.empty else pd.DataFrame()
            if not rules_fp.empty:
                rules_fp = rules_fp[rules_fp['lift'] >= min_lift]

            if not rules_ap.empty or not rules_fp.empty:
                rules = rules_ap if len(rules_ap) > len(rules_fp) else rules_fp
                algo_used = "Apriori" if len(rules_ap) > len(rules_fp) else "FP-Growth"
        else:
            if algorithm == "Apriori":
                frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
            else:
                frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                if not rules.empty:
                    rules = rules[rules['lift'] >= min_lift]
    except Exception as e:
        st.error(f"Mining error encountered: {str(e)}")
        
    return rules, algo_used

def get_llm_explanation(antecedents: str, consequents: str, support: float, confidence: float, lift: float) -> str:
    """Generate AI-powered or local fallback insights for association rules."""
    local_rules = {
        ('milk', 'bread'): "Breakfast essentials - often purchased together for morning meals",
        ('bread', 'butter'): "Classic combination for sandwiches and toast",
        ('milk', 'cheese'): "Dairy products frequently bought for family meals",
        ('eggs', 'bread'): "Perfect breakfast pair for making sandwiches",
        ('cheese', 'bread'): "Sandwich-making staples with high co-purchase rate",
        ('shampoo', 'conditioner'): "Hair care routine products - usually bought together",
        ('laptop', 'mouse'): "Tech accessories commonly purchased with computers",
        ('phone', 'case'): "Protection accessories bought with new devices",
        ('pizza', 'soda'): "Popular combo for meals and parties",
        ('burger', 'fries'): "Fast food items with strong association"
    }
    
    ant = antecedents.split(',')[0].strip()
    con = consequents.split(',')[0].strip()
    
    key = (ant.lower(), con.lower())
    if key in local_rules:
        return f"📈 {local_rules[key]} (Lift: {lift:.1f}x)"
    
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": f"Customers who buy {ant} often buy {con} with {confidence:.0%} confidence. Provide one sentence retail insight why."
                }],
                max_tokens=50,
                temperature=0.7
            )
            return f"🤖 {response.choices[0].message.content.strip()}"
        except Exception:
            return f"🔍 {ant}→{con}: Strong association with {lift:.1f}x lift"
    
    return f"📊 {ant}→{con}: {confidence:.0%} confidence • {lift:.1f}x lift"

def semantic_group_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Group rules into semantic categories based on item names."""
    rules_df = rules_df.copy()
    category_map = {
        'mil': 'Dairy', 'bre': 'Bakery', 'egg': 'Protein', 'che': 'Dairy',
        'but': 'Dairy', 'app': 'Fruits', 'ban': 'Fruits', 'yog': 'Dairy',
        'bac': 'Meat', 'cra': 'Snacks', 'sho': 'Apparel', 'tsh': 'Apparel',
        'jea': 'Apparel', 'sne': 'Footwear', 'bel': 'Accessories', 'soc': 'Accessories',
        'cap': 'Accessories', 'lap': 'Electronics', 'mou': 'Electronics', 'pho': 'Electronics',
        'cas': 'Accessories', 'sha': 'Personal Care', 'con': 'Personal Care'
    }
    
    rules_df['category'] = rules_df['antecedents'].apply(
        lambda x: category_map.get(convert_frozenset_to_string(x).split(',')[0][:3].lower(), 'General')
    )
    return rules_df

def validate_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate business validation scores for discovered rules."""
    sensible_rules = {
        'milk': ['bread', 'cheese', 'eggs', 'cereal', 'yogurt'],
        'bread': ['butter', 'milk', 'cheese', 'jam', 'eggs'],
        'eggs': ['bread', 'milk', 'bacon', 'cheese'],
        'cheese': ['bread', 'milk', 'crackers', 'wine'],
        'shampoo': ['conditioner', 'soap', 'lotion'],
        'laptop': ['mouse', 'bag', 'charger'],
        'phone': ['case', 'charger', 'screen protector'],
        'pizza': ['soda', 'garlic bread', 'wings'],
        'burger': ['fries', 'soda', 'shake']
    }
    
    scores = []
    for _, row in rules_df.iterrows():
        ant = convert_frozenset_to_string(row['antecedents']).split(',')[0].lower()
        con = convert_frozenset_to_string(row['consequents']).split(',')[0].lower()
        score = 1.0
        if ant in sensible_rules and con in sensible_rules[ant]:
            score = 1.3
        elif ant == con:
            score = 0.5
        scores.append(score)
    
    rules_df['business_score'] = scores
    return rules_df

def auto_suggest_params(df: pd.DataFrame) -> tuple[float, float]:
    """Smartly calculate default support and confidence based on dataset sparsity."""
    sparsity = df.sum().mean() / len(df)
    item_counts = df.sum()
    diversity = item_counts.std() / item_counts.mean()
    
    support = max(0.01, min(0.3, sparsity * 2))
    confidence = min(0.8, max(0.3, 0.6 - diversity * 0.2))
    return support, confidence

# ==========================================
# SAMPLE DATASETS
# ==========================================
sample_data = {
    "🛒 Grocery Store": """milk,bread,cheese,butter
bread,butter,eggs,milk
milk,cheese,yogurt,bread
apple,banana,bread,milk
eggs,milk,bacon,cheese
cheese,bread,crackers,wine
coffee,sugar,cream,milk
pasta,sauce,cheese,bread
cereal,milk,bread,butter
beer,chips,salsa,pretzels""",
    "👕 Fashion Retail": """shirt,jeans,sneakers,belt
jeans,belt,shirt,jacket
tshirt,jeans,cap,socks
shoes,socks,belt,shirt
dress,heels,jewelry,bag
jacket,scarf,gloves,hat
sunglasses,hat,bag,watch
hoodie,sweatpants,sneakers,socks
blazer,pants,dress_shoes,tie""",
    "📱 Electronics Store": """laptop,mouse,charger,bag
phone,case,charger,screen_protector
tablet,keyboard,case,stylus
headphones,phone,tablet,charger
tv,soundbar,hdrni_cables,mount
camera,lens,battery,memory_card
gaming_console,controller,games,headset
smartwatch,charger,bands,phone
router,cables,extender,adapter""",
    "💄 Beauty Products": """shampoo,conditioner,body_wash,lotion
face_wash,moisturizer,sunscreen,serum
lipstick,eyeliner,mascara,foundation
perfume,deodorant,body_spray,lotion
hair_dryer,brush,serum,shampoo
nail_polish,remover,cotton_pads,nail_file
makeup_brushes,cleanser,palette,primer
razor,shaving_cream,after_shave,moisturizer
face_mask,toner,cleanser,scrub""",
    "🏠 Home Improvement": """paint,brush,roller,tray
hammer,nails,screwdriver,screws
drill,bits,batteries,case
light_bulbs,fixture,wires,switch
gardening_gloves,seeds,soil,pots
tape_measure,level,utility_knife,blades
caulk,gun,putty_knife,sandpaper
extension_cord,power_strip,adapters,lights
ladder,bucket,cleaning_supplies,gloves""",
    "🍕 Fast Food Restaurant": """burger,fries,soda,shake
pizza,soda,garlic_bread,wings
sandwich,chips,cookie,drink
salad,dressing,breadsticks,soup
tacos,salsa,guacamole,drink
chicken_nuggets,fries,bbq_sauce,drink
breakfast_sandwich,hash_browns,coffee,juice
ice_cream,toppings,cup,cone
pretzel,cheese_sauce,drink,cookie""",
    "📚 Book Store": """fiction_book,bookmark,gift_card,mug
cookbook,apron,kitchen_tools,notebook
textbook,notebook,pens,highlighter
magazine,gift_card,coffee,snack
children_book,stuffed_toy,coloring_book,crayons
mystery_book,reading_light,bookstand,mug
biography,bookmark,notebook,pen
self_help,journal,pen,bookmark
art_book,sketchbook,pencils,erasers""",
    "🎮 Gaming Store": """console,controller,game,headset
game,strategy_guide,collectible,poster
gaming_chair,desk,headset,mouse
vr_headset,controllers,games,charger
trading_cards,binder,sleeves,deck_box
board_game,expansion,dice,timer
collectible_figure,display_case,lights,stand
gaming_keyboard,mouse,pad,headset
subscription_card,gift_card,merchandise,game""",
    "🏃 Sports Store": """running_shoes,socks,shorts,shirt
yoga_mat,blocks,strap,towel
basketball,hoop,net,pump
tennis_racket,balls,grip,bag
soccer_ball,cleats,shin_guards,socks
dumbbells,bench,gloves,strap
bicycle,helmet,lock,lights
swimsuit,goggles,cap,towel
hiking_boots,backpack,water_bottle,compass"""
}

# ==========================================
# UI LAYOUT & STATE INITIALIZATION
# ==========================================

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0;">📊 Smart Market Analyzer</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Discover hidden patterns in customer behavior with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    algorithm = st.selectbox("Choose mining method:", ["Apriori", "FP-Growth", "Both"], index=2, help="Select the frequent itemset mining algorithm.")
    
    st.markdown("---")
    st.markdown("#### 🤖 AI Features")
    if OPENAI_API_KEY:
        st.success("✅ AI Insights: Enabled")
        st.caption("GPT-powered rule explanations active")
    else:
        st.warning("⚠️ AI Insights: Local Only")
        st.caption("Using predefined fallback explanations")
    
    st.markdown("---")
    st.markdown("#### 📈 Visualizations")
    show_charts = st.checkbox("Show Interactive Charts", value=True)
    
    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.info("""
        This tool helps you discover:
        • What products are bought together
        • Customer buying patterns
        • Cross-selling opportunities
        • Market basket insights
        """)

# Initialize Session State
if 'mined_rules' not in st.session_state:
    st.session_state.mined_rules = None
if 'algo_used' not in st.session_state:
    st.session_state.algo_used = ""
if 'runtime' not in st.session_state:
    st.session_state.runtime = 0.0

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📥 Data Source")
    input_type = st.radio("Choose input method:", ["Sample Dataset", "Upload CSV"], horizontal=True, label_visibility="collapsed")

with col2:
    st.markdown("### &nbsp;")
    if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
        st.session_state.mined_rules = None
        st.session_state.algo_used = ""
        st.session_state.runtime = 0.0
        st.cache_data.clear()
        st.rerun()

transactions = None
if input_type == "Sample Dataset":
    st.markdown("#### Available Datasets")
    cols = st.columns(3)
    dataset_options = list(sample_data.keys())
    
    for i, dataset in enumerate(dataset_options):
        with cols[i % 3]:
            if st.button(f"**{dataset}**", use_container_width=True, help=f"Load {dataset} dataset"):
                st.session_state.selected_dataset = dataset
                st.session_state.mined_rules = None
    
    selected = st.session_state.get('selected_dataset', dataset_options[0])
    df_raw = pd.read_csv(StringIO(sample_data[selected]), header=None)
    transactions = df_raw.values.tolist()
    
    with st.expander(f"📋 Preview: {selected}", expanded=True):
        st.dataframe(df_raw, use_container_width=True)
        
elif input_type == "Upload CSV":
    uploaded = st.file_uploader("Upload your transaction data (CSV format)", type="csv")
    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded, header=None)
            if df_raw.empty or df_raw.shape[1] == 0:
                st.error("❌ The uploaded CSV file is empty or formatted incorrectly.")
                transactions = None
            else:
                transactions = df_raw.values.tolist()
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df_raw.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {str(e)}")

if transactions:
    progress_bar = st.progress(0, text="Preprocessing transaction data...")
    df_encoded = preprocess_transactions(transactions)
    progress_bar.progress(100, text="Preprocessing complete!")
    progress_bar.empty()
    
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Transactions", len(df_encoded))
    with col2:
        st.metric("Unique Items", df_encoded.shape[1])
    with col3:
        st.metric("Avg Items/Transaction", f"{df_encoded.sum().mean():.1f}")
    with col4:
        st.metric("Density", f"{(df_encoded.sum().sum() / (len(df_encoded) * df_encoded.shape[1])):.1%}")
    
    st.markdown("### ⚙️ Mining Parameters")
    supp, conf = auto_suggest_params(df_encoded)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            min_support = st.slider("Minimum Support", 0.01, 0.5, supp, 0.01, help="Frequency of item combination")
        with col1b:
            min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, conf, 0.05, help="Strength of association")
        with col1c:
            min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.2, 0.1, help="Interestingness factor")
    
    with col2:
        st.markdown("#### 🔍 Search Filter")
        nl_query = st.text_input("Filter rules by keyword:", placeholder="e.g., milk, bread...")
        if st.button("🚀 Start Mining", use_container_width=True, type="primary"):
            with st.spinner("⛏️ Mining association rules and calculating metrics..."):
                start_time = time.time()
                rules, algo_used = cached_mining(df_encoded, algorithm, min_support, min_confidence, min_lift)
                runtime = time.time() - start_time
                
                if not rules.empty:
                    rules = validate_rules(rules)
                    rules = semantic_group_rules(rules)
                
                st.session_state.mined_rules = rules
                st.session_state.algo_used = algo_used
                st.session_state.runtime = runtime

    # RENDER RESULTS FROM SESSION STATE
    if st.session_state.mined_rules is not None:
        rules = st.session_state.mined_rules.copy()
        algo_used = st.session_state.algo_used
        runtime = st.session_state.runtime
        
        if nl_query and not rules.empty:
            mask = (
                rules['antecedents'].astype(str).str.contains(nl_query, case=False, na=False) |
                rules['consequents'].astype(str).str.contains(nl_query, case=False, na=False)
            )
            rules = rules[mask]

        if rules.empty:
            st.warning("⚠️ No rules match your criteria or keyword filter.")
        else:
            st.markdown(f"""
            <div style="background: #f0f7ff; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin: 0; color: #1e3a8a;">✅ Mining Complete!</h3>
                <p style="margin: 0.5rem 0 0 0; color: #4b5563;">
                    Found <strong>{len(rules)}</strong> association rules using <strong>{algo_used}</strong> in <strong>{runtime:.2f}s</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🎯 Discovered Rules")
            rules_display = rules.copy()
            for col in ['antecedents', 'consequents']:
                rules_display[col] = rules_display[col].apply(convert_frozenset_to_string)

            explanations = []
            for idx, row in rules.head(20).iterrows():
                exp = get_llm_explanation(
                    convert_frozenset_to_string(row['antecedents']),
                    convert_frozenset_to_string(row['consequents']),
                    row['support'],
                    row['confidence'],
                    row['lift']
                )
                explanations.append(exp)

            if len(rules_display) > len(explanations):
                explanations.extend(["See stats for details"] * (len(rules_display) - len(explanations)))
            
            rules_display['ai_insight'] = explanations

            display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'business_score', 'category', 'ai_insight']
            available_cols = [col for col in display_cols if col in rules_display.columns]

            st.dataframe(
                rules_display[available_cols],
                column_config={
                    "antecedents": "If bought...",
                    "consequents": "Then also buy...",
                    "support": st.column_config.NumberColumn("Support", format="%.3f"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.1%"),
                    "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
                    "business_score": st.column_config.NumberColumn("Biz Score", format="%.2f"),
                    "category": "Category",
                    "ai_insight": "AI Insight"
                },
                use_container_width=True,
                height=400
            )

            if show_charts and len(rules) > 1:
                st.markdown("### 📈 Advanced Analytics")
                rules_plotly = safe_df_for_plotly(rules)

                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.scatter(
                        rules_plotly.head(30), x='support', y='confidence', size='lift', color='business_score',
                        hover_name='antecedents', hover_data=['consequents', 'lift'],
                        title="📊 Rule Quality Matrix", labels={'support': 'Support', 'confidence': 'Confidence'},
                        color_continuous_scale='viridis'
                    )
                    fig1.update_layout(height=400)
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    top_rules = rules.nlargest(10, 'lift')
                    top_safe = safe_df_for_plotly(top_rules)
                    fig2 = px.bar(
                        top_safe, x='lift', y='antecedents', color='confidence', orientation='h',
                        hover_data=['consequents', 'support'], title="🏆 Top 10 Rules by Lift",
                        labels={'lift': 'Lift Value', 'antecedents': 'If bought...'}, color_continuous_scale='plasma'
                    )
                    fig2.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig2, use_container_width=True)

                col3, col4 = st.columns(2)
                with col3:
                    if 'category' in rules_plotly.columns:
                        cat_counts = rules_plotly['category'].value_counts().reset_index()
                        cat_counts.columns = ['Category', 'Count']
                        fig3 = px.pie(cat_counts, values='Count', names='Category', title="📦 Rule Categories", hole=0.4)
                        fig3.update_layout(height=400)
                        st.plotly_chart(fig3, use_container_width=True)

                with col4:
                    fig4 = px.scatter(
                        rules_plotly, x='confidence', y='business_score', color='lift', size='support',
                        hover_name='antecedents', title="💼 Business Value Analysis",
                        labels={'confidence': 'Confidence', 'business_score': 'Business Score'},
                        color_continuous_scale='RdYlGn'
                    )
                    fig4.update_layout(height=400)
                    st.plotly_chart(fig4, use_container_width=True)

            st.markdown("### 📥 Export Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv_data = rules.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download CSV", csv_data, "association_rules.csv", "text/csv", use_container_width=True)
            with col2:
                summary = {
                    "total_rules": len(rules),
                    "best_lift": float(rules['lift'].max()) if not rules.empty else 0.0,
                    "avg_confidence": float(rules['confidence'].mean()) if not rules.empty else 0.0,
                    "algorithm_used": algo_used,
                    "runtime_seconds": runtime
                }
                st.download_button("📊 Summary JSON", str(summary).encode('utf-8'), "analysis_summary.json", "application/json", use_container_width=True)
            with col3:
                if st.button("🔄 Clear Results", use_container_width=True):
                    st.session_state.mined_rules = None
                    st.rerun()
else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
        <h3 style="color: #4b5563; margin-bottom: 1rem;">🚀 Ready to Discover Insights</h3>
        <p style="color: #6b7280; max-width: 600px; margin: 0 auto;">
            Select a sample dataset from our collection or upload your transaction data to uncover hidden shopping patterns and boost business intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">
<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <p style="margin: 0.5rem 0;">Uncover hidden customer behaviors • Optimize product placement • Increase cross-selling revenue</p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">© 2026 Smart Market Analyzer</p>
</div>
""", unsafe_allow_html=True)