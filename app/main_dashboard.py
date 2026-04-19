import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor

# --- UI CONFIGURATION (Premium Look) ---
st.set_page_config(page_title="Retail Forecasting OS", layout="wide", page_icon="📈")

# Custom CSS for Premium Visuals
st.markdown("""
    <style>
    .kpi-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .kpi-title { color: #6c757d; font-size: 14px; font-weight: bold; text-transform: uppercase;}
    .kpi-value { color: #212529; font-size: 28px; font-weight: bold; margin: 10px 0;}
    .kpi-trend-up { color: #28a745; font-size: 14px; font-weight: bold;}
    .kpi-trend-down { color: #dc3545; font-size: 14px; font-weight: bold;}
    .alert-box { padding: 15px; border-radius: 5px; margin-bottom: 15px; font-weight: bold; }
    .alert-critical { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_z_score(service_level):
    sl_map = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
    return sl_map.get(service_level, 1.65)

@st.cache_data
def load_and_validate_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Enterprise Validation: Ensure the user's CSV won't break our math
            required_cols = ['Date', 'Product_ID', 'Sales']
            if not all(col in df.columns for col in required_cols):
                st.sidebar.error(f"⚠ Missing columns! Your CSV must contain exact headers: {required_cols}")
                return None
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            return None
    else:
        # Fallback to our synthetic raw_sales.csv if no file is uploaded
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '..', 'data', 'raw_sales.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        return None

# --- SIDEBAR: SYSTEM CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2800/2800230.png", width=50) # Mock Logo
    st.header("⚙️ Supply Chain OS")
    
    st.markdown("### 📂 Data Source")
    user_file = st.file_uploader("Upload custom CSV", type=['csv'])
    
    df = load_and_validate_data(user_file)
    
    if df is None:
        st.error("Data missing. Please upload a CSV or run data_loader.py")
        st.stop()
        
    st.markdown("### 🔍 Filters")
    selected_product = st.selectbox("Select SKU", df['Product_ID'].unique())
    granularity = st.radio("Granularity", ["Daily", "Weekly"], horizontal=True)
    
    st.markdown("### 🔄 Scenario Simulation")
    lead_time = st.slider("Supplier Lead Time (Days)", 1, 14, 3)
    service_level_str = st.selectbox("Target Service Level", ["90%", "95%", "99%"], index=1)
    demand_shock = st.slider("Simulate Demand Shock (%)", -50, 100, 0, step=10)

# --- DATA PROCESSING ---
prod_df = df[df['Product_ID'] == selected_product].sort_values('Date').copy()

if granularity == "Weekly":
    prod_df = prod_df.resample('W', on='Date').sum().reset_index()

# Apply Simulation Shock
prod_df['Sales'] = prod_df['Sales'] * (1 + (demand_shock/100))

# Calculations
recent_sales = prod_df['Sales'].tail(30).mean() if granularity == "Daily" else prod_df['Sales'].tail(4).mean()
std_dev = prod_df['Sales'].std()
z_score = get_z_score(service_level_str)

# Advanced Inventory Math
safety_stock = int(z_score * std_dev * (lead_time ** 0.5))
reorder_point = int((recent_sales * lead_time) + safety_stock)

# --- MAIN LAYOUT ---
st.title("📊 Retail Sales Forecasting & Inventory Optimization System")
st.markdown(f"**Target SKU:** `{selected_product}` &nbsp;&nbsp;|&nbsp;&nbsp; **View:** `{granularity}` &nbsp;&nbsp;|&nbsp;&nbsp; **Service Level Target:** `{service_level_str}`")
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "📈 Advanced Forecasting", "🧮 Inventory & Costs"])

# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tab1:
    # PREMIUM KPI CARDS
    c1, c2, c3, c4 = st.columns(4)
    trend = ((recent_sales - prod_df['Sales'].iloc[-60:-30].mean()) / prod_df['Sales'].iloc[-60:-30].mean()) * 100
    trend_class = "kpi-trend-up" if trend > 0 else "kpi-trend-down"
    trend_arrow = "↑" if trend > 0 else "↓"

    c1.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Demand</div>
            <div class="kpi-value">{int(recent_sales)}</div>
            <div class="{trend_class}">{trend_arrow} {abs(trend):.1f}% vs last period</div>
        </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Reorder Point (ROP)</div>
            <div class="kpi-value">{reorder_point}</div>
            <div class="kpi-title" style="color: #1f77b4;">Lead Time: {lead_time} Days</div>
        </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Safety Stock</div>
            <div class="kpi-value">{safety_stock}</div>
            <div class="kpi-title" style="color: #ff7f0e;">Risk Buffer</div>
        </div>
    """, unsafe_allow_html=True)

    # DECISION INSIGHT ENGINE
    st.subheader("🤖 AI Business Insights")
    if trend > 10:
        st.success(f"**Growth Alert:** Demand is surging ({trend:.1f}%). Consider increasing base inventory levels to prevent impending stockouts.")
    if demand_shock > 0:
        st.warning(f"**Simulation Active:** System calculating parameters based on a {demand_shock}% artificial demand spike.")
        
    # ANOMALY DETECTION
    prod_df['Z_Score'] = (prod_df['Sales'] - prod_df['Sales'].mean()) / prod_df['Sales'].std()
    anomalies = prod_df[abs(prod_df['Z_Score']) > 2.5]
    
    fig = px.line(prod_df, x='Date', y='Sales', title="Historical Sales with Anomaly Detection", template="plotly_white")
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Sales'], mode='markers', 
                                 marker=dict(color='red', size=10), name="Anomalies"))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: ADVANCED FORECASTING
# ==========================================
with tab2:
    st.subheader("Machine Learning Demand Forecast")
    
    # Simple Feature Eng & Model
    train = prod_df.copy()
    train['lag1'] = train['Sales'].shift(1)
    train = train.dropna()
    
    X = train[['lag1']]
    y = train['Sales']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Predict next 14 periods
    future_dates = [train['Date'].max() + timedelta(days=i*(7 if granularity=='Weekly' else 1)) for i in range(1, 15)]
    curr_lag = y.iloc[-1]
    preds = []
    for _ in range(14):
        p = model.predict([[curr_lag]])[0]
        preds.append(p)
        curr_lag = p
        
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': preds})
    
    # CONFIDENCE INTERVALS
    error_margin = std_dev * 0.8 # Simulated model error
    forecast_df['Upper'] = forecast_df['Forecast'] + error_margin
    forecast_df['Lower'] = np.maximum(0, forecast_df['Forecast'] - error_margin) # Can't have negative sales

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train['Date'].tail(30), y=train['Sales'].tail(30), name="Actual", line=dict(color="blue")))
    fig2.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name="Forecast", line=dict(color="orange", dash="dash")))
    
    # Shaded Area
    fig2.add_trace(go.Scatter(x=pd.concat([forecast_df['Date'], forecast_df['Date'][::-1]]),
                              y=pd.concat([forecast_df['Upper'], forecast_df['Lower'][::-1]]),
                              fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                              name="95% Confidence Interval"))
    fig2.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 3: INVENTORY & COSTS
# ==========================================
with tab3:
    col_g, col_rec = st.columns([1, 1.5])
    
    # Live Stock Input
    current_stock = st.sidebar.number_input("Input Current Warehouse Stock", value=int(reorder_point * 1.2))

    with col_g:
        # STOCK RISK VISUALIZATION (Gauge)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_stock,
            title = {'text': "Current Stock Health"},
            gauge = {
                'axis': {'range': [0, reorder_point * 2]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, safety_stock], 'color': "red"},
                    {'range': [safety_stock, reorder_point], 'color': "yellow"},
                    {'range': [reorder_point, reorder_point * 2], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': reorder_point
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_rec:
        # ACTIONABLE RECOMMENDATIONS
        st.subheader("🚀 Strategic Action Plan")
        if current_stock <= safety_stock:
            st.markdown(f'<div class="alert-box alert-critical">CRITICAL RISK: Stock is in the red zone. Order {int((reorder_point * 1.5) - current_stock)} units via EXPEDITED shipping today.</div>', unsafe_allow_html=True)
        elif current_stock <= reorder_point:
            st.warning(f"ACTION REQUIRED: Stock has breached Reorder Point. Order {int((reorder_point * 1.2) - current_stock)} units standard delivery.")
        else:
            st.success("✅ Stock levels are healthy. No immediate purchasing action required.")
            
        st.divider()
        
        # COST OPTIMIZATION (Mock Financials)
        st.markdown("### 💰 Financial Impact Analysis")
        holding_cost_per_unit = 2.50
        stockout_cost_per_unit = 45.00
        
        st.write(f"- **Est. Monthly Holding Cost:** ${int(current_stock * holding_cost_per_unit):,}")
        if current_stock < safety_stock:
            risk_units = safety_stock - current_stock
            st.write(f"- **Potential Stockout Risk Cost:** <span style='color:red'>${int(risk_units * stockout_cost_per_unit):,}</span>", unsafe_allow_html=True)
        else:
            st.write(f"- **Potential Stockout Risk Cost:** $0 (Adequately buffered)")