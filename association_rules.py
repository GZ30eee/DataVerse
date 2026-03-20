import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import time
import openai
import os
from collections import Counter

# Page config
st.set_page_config(page_title="📊 Smart Market Analyzer", layout="wide", page_icon="🛍️")

OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")  # Loaded from secrets.toml

# Initialize OpenAI if key is provided
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# CRITICAL FIXES
def convert_frozenset_to_string(obj):
    """Convert frozenset to string - 100% safe"""
    if isinstance(obj, frozenset):
        return ', '.join(sorted(list(obj)))
    return str(obj)

def safe_df_for_plotly(df):
    """Convert ALL frozensets to strings for Plotly JSON"""
    df_safe = df.copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == 'object':
            df_safe[col] = df_safe[col].apply(lambda x: convert_frozenset_to_string(x))
    return df_safe

# 🔥 LLM EXPLANATIONS
def get_llm_explanation(antecedents, consequents, support, confidence, lift):
    """🤖 AI-Powered Insights"""
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
    
    # Use OpenAI for insights
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
        except:
            return f"🔍 {ant}→{con}: Strong association with {lift:.1f}x lift"
    
    return f"📊 {ant}→{con}: {confidence:.0%} confidence • {lift:.1f}x lift"

def semantic_group_rules(rules_df):
    """📊 Group rules by category"""
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

def validate_rules(rules_df):
    """✅ Business Logic Validation"""
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
            score = 1.3  # Bonus for sensible combinations
        elif ant == con:
            score = 0.5  # Penalty for self-association
        scores.append(score)
    
    rules_df['business_score'] = scores
    return rules_df

def auto_suggest_params(df):
    """✨ Smart Parameter Suggestions"""
    sparsity = df.sum().mean() / len(df)
    item_counts = df.sum()
    diversity = item_counts.std() / item_counts.mean()
    
    support = max(0.05, min(0.3, sparsity * 2))
    confidence = min(0.8, max(0.4, 0.6 - diversity * 0.2))
    
    return support, confidence

# 🎯 EXTENSIVE SAMPLE DATASETS (9 datasets)
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

# MAIN APP - Clean Header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0;">📊 Smart Market Analyzer</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Discover hidden patterns in customer behavior with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

# SIMPLIFIED SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### Algorithm")
    algorithm = st.selectbox("Choose mining method:", ["Apriori", "FP-Growth", "Both"], index=2)
    
    st.markdown("---")
    st.markdown("#### 🤖 AI Features")
    if OPENAI_API_KEY:
        st.success("✅ AI Insights: Enabled")
        st.caption("GPT-powered rule explanations")
    else:
        st.warning("⚠️ AI Insights: Local Only")
        st.caption("Using predefined explanations")
    
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

# DATA SELECTION - Improved Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📥 Data Source")
    input_type = st.radio("Choose input method:", ["Sample Dataset", "Upload CSV"], horizontal=True, label_visibility="collapsed")

with col2:
    st.markdown("### &nbsp;")
    if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
        st.rerun()

# Load data
transactions = None
if input_type == "Sample Dataset":
    st.markdown("#### Available Datasets")
    # Create 3 columns for dataset selection
    cols = st.columns(3)
    dataset_options = list(sample_data.keys())
    
    for i, dataset in enumerate(dataset_options):
        with cols[i % 3]:
            if st.button(f"**{dataset}**", use_container_width=True, 
                        help=f"Load {dataset} dataset"):
                st.session_state.selected_dataset = dataset
    
    # Display selected dataset
    selected = st.session_state.get('selected_dataset', dataset_options[0])
    df_raw = pd.read_csv(StringIO(sample_data[selected]), header=None)
    transactions = df_raw.values.tolist()
    
    with st.expander(f"📋 Preview: {selected}", expanded=True):
        st.dataframe(df_raw, use_container_width=True)
        
elif input_type == "Upload CSV":
    uploaded = st.file_uploader("Upload your transaction data (CSV format)", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded, header=None)
        transactions = df_raw.values.tolist()
        
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df_raw.head(20), use_container_width=True)

# PROCESSING SECTION
if transactions:
    with st.spinner("🔄 Processing transactions..."):
        df = pd.DataFrame(transactions).stack().str.get_dummies().groupby(level=0).sum().astype(bool)
    
    # Stats overview in nice cards
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Transactions", len(df))
    with col2:
        st.metric("Unique Items", df.shape[1])
    with col3:
        st.metric("Avg Items/Transaction", f"{df.sum().mean():.1f}")
    with col4:
        st.metric("Density", f"{(df.sum().sum() / (len(df) * df.shape[1])):.1%}")
    
    # PARAMETERS SECTION - Cleaner
    st.markdown("### ⚙️ Mining Parameters")
    
    supp, conf = auto_suggest_params(df)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            min_support = st.slider("Minimum Support", 0.01, 0.5, supp, 0.01, 
                                   help="Frequency of item combination")
        with col1b:
            min_confidence = st.slider("Minimum Confidence", 0.3, 1.0, conf, 0.05,
                                      help="Strength of association")
        with col1c:
            min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.2, 0.1,
                                help="Interestingness factor")
    
    with col2:
        st.markdown("#### 🔍 Search Filter")
        nl_query = st.text_input("Filter rules by keyword:", placeholder="e.g., milk, bread...")
        if st.button("🚀 Start Mining", use_container_width=True, type="primary"):
            st.session_state.run_mining = True
    
    # MINING EXECUTION
    if st.session_state.get('run_mining', False):
        with st.spinner("⛏️ Mining association rules..."):
            start_time = time.time()
            
            # Run selected algorithm
            if algorithm == "Both":
                # Apriori
                freq_ap = apriori(df, min_support=min_support, use_colnames=True)
                if not freq_ap.empty:
                    rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=min_confidence)
                    rules_ap = rules_ap[rules_ap['lift'] >= min_lift]
                else:
                    rules_ap = pd.DataFrame()
                    st.info("Apriori found no frequent itemsets with current support.")

                # FP-Growth
                freq_fp = fpgrowth(df, min_support=min_support, use_colnames=True)
                if not freq_fp.empty:
                    rules_fp = association_rules(freq_fp, metric="confidence", min_threshold=min_confidence)
                    rules_fp = rules_fp[rules_fp['lift'] >= min_lift]
                else:
                    rules_fp = pd.DataFrame()
                    st.info("FP-Growth found no frequent itemsets with current support.")

                # Choose the better result
                if not rules_ap.empty or not rules_fp.empty:
                    rules = rules_ap if len(rules_ap) > len(rules_fp) else rules_fp
                    algo_used = "Apriori" if len(rules_ap) > len(rules_fp) else "FP-Growth"
                else:
                    rules = pd.DataFrame()
                    algo_used = "None"
            else:
                if algorithm == "Apriori":
                    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
                else:
                    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

                if not frequent_itemsets.empty:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                    rules = rules[rules['lift'] >= min_lift]
                    algo_used = algorithm
                else:
                    rules = pd.DataFrame()
                    algo_used = algorithm
                    st.warning("No frequent itemsets found. Try lowering the minimum support.")
            runtime = time.time() - start_time

            if not rules.empty:
                # ✅ Apply enhancements only when rules exist
                rules = validate_rules(rules)
                rules = semantic_group_rules(rules)

                # Apply natural language filter if provided
                if nl_query:
                    mask = (
                        rules['antecedents'].astype(str).str.contains(nl_query, case=False, na=False) |
                        rules['consequents'].astype(str).str.contains(nl_query, case=False, na=False)
                    )
                    rules = rules[mask]

                # If filter empties the set, show a warning and skip further display
                if rules.empty:
                    st.warning("No rules match your keyword filter.")
                else:
                    # RESULTS HEADER
                    st.markdown(f"""
                    <div style="background: #f0f7ff; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="margin: 0; color: #1e3a8a;">✅ Mining Complete!</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #4b5563;">
                            Found <strong>{len(rules)}</strong> association rules using <strong>{algo_used}</strong> in <strong>{runtime:.2f}s</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # MAIN RESULTS - Rules with AI Insights
                    st.markdown("### 🎯 Discovered Rules")

                    # Prepare rules for display
                    rules_display = rules.copy()
                    for col in ['antecedents', 'consequents']:
                        rules_display[col] = rules_display[col].apply(convert_frozenset_to_string)

                    # Add AI explanations (limit to top 20 for quality)
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

                    # Ensure the AI insights column has the correct length by padding
                    if len(rules_display) > len(explanations):
                        explanations.extend(["See stats for details"] * (len(rules_display) - len(explanations)))
                    
                    rules_display['ai_insight'] = explanations

                    # Display rules in a clean table
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

                    # INTERACTIVE VISUALIZATIONS
                    if show_charts and len(rules) > 1:
                        st.markdown("### 📈 Advanced Analytics")

                        # Prepare data for plotting
                        rules_plotly = safe_df_for_plotly(rules)

                        # Create 2x2 grid of charts
                        col1, col2 = st.columns(2)

                        with col1:
                            # Chart 1: Scatter plot
                            fig1 = px.scatter(
                                rules_plotly.head(30),
                                x='support',
                                y='confidence',
                                size='lift',
                                color='business_score',
                                hover_name='antecedents',
                                hover_data=['consequents', 'lift'],
                                title="📊 Rule Quality Matrix",
                                labels={'support': 'Support', 'confidence': 'Confidence'},
                                color_continuous_scale='viridis'
                            )
                            fig1.update_layout(height=400)
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            # Chart 2: Top rules by lift
                            top_rules = rules.nlargest(10, 'lift')
                            top_safe = safe_df_for_plotly(top_rules)
                            fig2 = px.bar(
                                top_safe,
                                x='lift',
                                y='antecedents',
                                color='confidence',
                                orientation='h',
                                hover_data=['consequents', 'support'],
                                title="🏆 Top 10 Rules by Lift",
                                labels={'lift': 'Lift Value', 'antecedents': 'If bought...'},
                                color_continuous_scale='plasma'
                            )
                            fig2.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig2, use_container_width=True)

                        col3, col4 = st.columns(2)

                        with col3:
                            # Chart 3: Category distribution
                            if 'category' in rules_plotly.columns:
                                cat_counts = rules_plotly['category'].value_counts().reset_index()
                                cat_counts.columns = ['Category', 'Count']
                                fig3 = px.pie(
                                    cat_counts,
                                    values='Count',
                                    names='Category',
                                    title="📦 Rule Categories",
                                    hole=0.4
                                )
                                fig3.update_layout(height=400)
                                st.plotly_chart(fig3, use_container_width=True)

                        with col4:
                            # Chart 4: Confidence vs Business Score
                            fig4 = px.scatter(
                                rules_plotly,
                                x='confidence',
                                y='business_score',
                                color='lift',
                                size='support',
                                hover_name='antecedents',
                                title="💼 Business Value Analysis",
                                labels={'confidence': 'Confidence', 'business_score': 'Business Score'},
                                color_continuous_scale='RdYlGn'
                            )
                            fig4.update_layout(height=400)
                            st.plotly_chart(fig4, use_container_width=True)

                        # Chart 5: Parallel coordinates
                        if len(rules) >= 5:
                            st.markdown("#### 🔗 Multi-dimensional Analysis")
                            fig5 = px.parallel_coordinates(
                                rules_plotly[['support', 'confidence', 'lift', 'business_score']].head(20),
                                color='lift',
                                labels={'support': 'Support', 'confidence': 'Confidence',
                                       'lift': 'Lift', 'business_score': 'Business Score'},
                                color_continuous_scale=px.colors.diverging.Tealrose,
                                title="Multi-dimensional Rule Comparison"
                            )
                            st.plotly_chart(fig5, use_container_width=True)

                    # DOWNLOAD SECTION
                    st.markdown("### 📥 Export Results")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        csv_data = rules.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📄 Download CSV",
                            csv_data,
                            "association_rules.csv",
                            "text/csv",
                            use_container_width=True
                        )

                    with col2:
                        # Summary JSON
                        summary = {
                            "total_rules": len(rules),
                            "best_lift": float(rules['lift'].max()),
                            "avg_confidence": float(rules['confidence'].mean()),
                            "algorithm_used": algo_used,
                            "runtime_seconds": runtime
                        }
                        st.download_button(
                            "📊 Summary JSON",
                            str(summary).encode('utf-8'),
                            "analysis_summary.json",
                            "application/json",
                            use_container_width=True
                        )

                    with col3:
                        if st.button("🔄 New Analysis", use_container_width=True):
                            st.session_state.run_mining = False
                            st.rerun()

            else:
                st.warning("⚠️ No association rules found with current parameters. Try lowering support or confidence thresholds.")
                if st.button("🔄 Adjust Parameters", use_container_width=True):
                    st.session_state.run_mining = False            
            

else:
    # WELCOME STATE
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
        <h3 style="color: #4b5563; margin-bottom: 1rem;">🚀 Ready to Discover Insights</h3>
        <p style="color: #6b7280; max-width: 600px; margin: 0 auto;">
            Select a sample dataset from our extensive collection or upload your own transaction data 
            to uncover hidden patterns and boost your business intelligence.
        </p>
        <div style="margin-top: 2rem; color: #9ca3af; font-size: 0.9rem;">
            <p>✨ Features included:</p>
            <div style="display: inline-flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin-top: 1rem;">
                <span>🤖 AI-Powered Insights</span> • 
                <span>📊 5 Interactive Charts</span> • 
                <span>🎯 Smart Recommendations</span> • 
                <span>⚡ Real-time Analysis</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ENHANCED FOOTER
st.markdown("""
<hr style="margin: 3rem 0 1rem 0;">

<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
        <span>🔍 <strong>Pattern Discovery</strong></span>
        <span>🤖 <strong>AI Intelligence</strong></span>
        <span>📈 <strong>Visual Analytics</strong></span>
        <span>⚡ <strong>Real-time Processing</strong></span>
    </div>
    <p style="margin: 0.5rem 0;">
        Uncover hidden customer behaviors • Optimize product placement • Increase cross-selling revenue
    </p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">
        © 2024 Smart Market Analyzer | Built for data-driven decision making
    </p>
</div>
""", unsafe_allow_html=True)