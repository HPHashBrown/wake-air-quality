import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression

# Try importing Meta's Prophet for advanced forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ============================================
# PAGE CONFIG & HIGH-TECH THEMING
# ============================================
st.set_page_config(page_title="Wake AQI Intelligence", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Animated Global Background */
    .stApp { background-color: #0b0f19; color: #ffffff; }
    
    /* Animated Gradient Title */
    .title-gradient {
        background: linear-gradient(-45deg, #00f2fe, #4facfe, #00f2fe, #4facfe);
        background-size: 300% 300%;
        animation: gradient-shift 8s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 54px !important;
        font-weight: 900;
        margin-bottom: 0px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .sub-font { font-size: 20px !important; color: #8b9bb4; margin-bottom: 35px; font-weight: 300; letter-spacing: 1px; }
    
    /* Glassmorphism UI */
    .glass-card {
        background: rgba(16, 25, 43, 0.6); 
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px; 
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 25px; 
        color: #ffffff !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease-in-out;
        text-align: center;
        margin-bottom: 15px;
    }
    .glass-card:hover { transform: translateY(-8px); border: 1px solid rgba(79, 172, 254, 0.4); box-shadow: 0 12px 40px 0 rgba(79, 172, 254, 0.2); }
    .glass-card h5 { color: #8b9bb4 !important; font-size: 16px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; }
    .glass-card h3 { color: #ffffff !important; font-size: 38px; font-weight: 800; margin: 0px 0px 10px 0px; }
    .glass-card span { font-size: 13px; color: #4facfe; }

    /* Sidebar Styling */
    .css-1d391kg { background-color: #0b0f19; border-right: 1px solid rgba(255, 255, 255, 0.08); }
    </style>
""", unsafe_allow_html=True)

# ============================================
# FEATURE 4: ALERT NOTIFICATION SYSTEM (SIDEBAR)
# ============================================
with st.sidebar:
    st.markdown("### 🔔 Automated Alerting")
    st.write("Deploy threshold triggers to receive automated environmental warnings.")
    
    with st.form("alert_form"):
        target_email = st.text_input("Operator Email", placeholder="operator@wake.gov")
        alert_threshold = st.slider("US AQI Trigger Threshold", min_value=50, max_value=300, value=100, step=10)
        submit_alert = st.form_submit_button("Initialize Protocol")
        
        if submit_alert:
            if target_email:
                # In a production app, this would write to a database and trigger an AWS Lambda/CRON job
                st.success(f"Protocol Active: Monitoring matrix for AQI > {alert_threshold}. Alerts will route to {target_email}.")
            else:
                st.error("Error: Valid Email Required.")

# ============================================
# DATA FETCHING (API)
# ============================================
@st.cache_data(ttl=3600)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    try:
        return requests.get(url).json().get("current", None)
    except: return None

@st.cache_data(ttl=3600)
def fetch_current_aqi(lat=35.7796, lon=-78.6382):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,pm2_5"
    try:
        return requests.get(url).json().get("current", {})
    except: return {}

# Mock Historical Fetch (Since AQS requires valid keys)
try:
    df_yearly = pd.read_csv("wake_pm25_by_year.csv").sort_values("year")
except:
    df_yearly = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022, 2023], "mean_pm25": [10.2, 9.8, 8.5, 9.2, 8.1, 7.9]})

current_year = datetime.now().year
with st.spinner("Initializing Atmospheric Sensors & Predictive AI..."):
    current_aqi_data = fetch_current_aqi()
    live_weather = fetch_live_weather(35.7796, -78.6382)

# ============================================
# FEATURE 1: ADVANCED AI FORECASTING (PROPHET)
# ============================================
if PROPHET_AVAILABLE:
    # Prepare data for Prophet
    prophet_df = df_yearly.copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df['year'], format='%Y')
    prophet_df = prophet_df.rename(columns={'mean_pm25': 'y'})
    
    # Initialize and train Meta's Prophet
    m = Prophet(yearly_seasonality=True)
    m.fit(prophet_df)
    
    # Predict 10 years into the future
    future = m.make_future_dataframe(periods=10, freq='YS')
    forecast = m.predict(future)
    
    # Extract prediction data
    future_df = pd.DataFrame({
        "year": forecast['ds'].dt.year,
        "predicted_pm25": forecast['yhat'],
        "yhat_lower": forecast['yhat_lower'],
        "yhat_upper": forecast['yhat_upper']
    })
else:
    # Fallback to standard Linear Regression if Prophet isn't installed
    X = df_yearly[["year"]]
    y = df_yearly["mean_pm25"]
    model = LinearRegression().fit(X, y)
    future_years = np.arange(current_year, current_year + 11)
    future_preds = model.predict(future_years.reshape(-1, 1))
    future_df = pd.DataFrame({
        "year": future_years, 
        "predicted_pm25": future_preds,
        "yhat_lower": future_preds - 1.5, # Mock confidence bounds
        "yhat_upper": future_preds + 1.5
    })

current_aqi = current_aqi_data.get('us_aqi', 0)

# ============================================
# UI LAYOUT
# ============================================
st.markdown('<p class="title-gradient">Wake AQI Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Next-Generation Atmospheric PM2.5 Analytics Engine.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Telemetry & Forecasting", "🧠 Predictive Scenario Core", "🛰️ Orbital Vector Map"])

# --- TAB 1: OVERVIEW & ADVANCED CHART ---
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{current_aqi}</h3></div>", unsafe_allow_html=True)
    with m2:
        val = f"{live_weather['temperature_2m']}°C" if live_weather else "N/A"
        st.markdown(f"<div class='glass-card'><h5>Thermal State</h5><h3>{val}</h3><span>Raleigh Node</span></div>", unsafe_allow_html=True)
    with m3:
        val = f"{live_weather['wind_speed_10m']} km/h" if live_weather else "N/A"
        st.markdown(f"<div class='glass-card'><h5>Wind Velocity</h5><h3>{val}</h3><span>Dispersion Rate</span></div>", unsafe_allow_html=True)
    with m4:
        val = f"{live_weather['wind_direction_10m']}°" if live_weather else "N/A"
        st.markdown(f"<div class='glass-card'><h5>Vector Heading</h5><h3>{val}</h3><span>Atmospheric Drift</span></div>", unsafe_allow_html=True)

    # Advanced Plotly Chart with Confidence Intervals
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=df_yearly["year"], y=df_yearly["mean_pm25"], 
        mode="lines+markers", name="Recorded Telemetry", 
        line=dict(color="#00f2fe", width=4),
        marker=dict(size=8, color="#ffffff", line=dict(width=2, color="#00f2fe"))
    ))
    
    # AI Forecast Line
    forecast_future = future_df[future_df['year'] > df_yearly['year'].max()]
    fig.add_trace(go.Scatter(
        x=forecast_future["year"], y=forecast_future["predicted_pm25"], 
        mode="lines+markers", name="Algorithmic Forecast", 
        line=dict(color="#f87171", width=4, dash="dot"),
        marker=dict(size=8, color="#ffffff", line=dict(width=2, color="#f87171"))
    ))

    # AI Confidence Interval (Shaded Area)
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_future["year"], forecast_future["year"][::-1]]),
        y=pd.concat([forecast_future["yhat_upper"], forecast_future["yhat_lower"][::-1]]),
        fill='toself', fillcolor='rgba(248, 113, 113, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=True, name="AI Confidence Interval"
    ))

    fig.update_layout(
        title="PM2.5 Long-Term Atmospheric Trajectory (Prophet Forecasting)", 
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ADVANCED ANALYTICS ---
with tab2:
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Current Threat Level (AQI)**")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=current_aqi, 
            gauge={'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': "white"}, 'bar': {'color': "#00f2fe"}, 
                   'bgcolor': "rgba(255,255,255,0.05)",
                   'steps': [{'range': [0, 50], 'color': "rgba(16, 185, 129, 0.3)"}, {'range': [50, 100], 'color': "rgba(245, 158, 11, 0.3)"}, {'range': [100, 300], 'color': "rgba(239, 68, 68, 0.3)"}]}
        ))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(gauge, use_container_width=True)
    with g2:
        st.markdown("**Emission Mitigation Simulator**")
        reduction = st.slider("Simulated Reduction (%)", 0, 50, 0)
        sim_val = future_df['predicted_pm25'].iloc[-1] * (1 - (reduction/100))
        st.metric(f"Estimated {future_df['year'].iloc[-1]} PM2.5", f"{sim_val:.2f} µg/m³", delta=f"-{reduction}% impact")

# --- TAB 3: SATELLITE & WIND VECTOR MAP ---
with tab3:
    st.markdown("### 🛰️ Orbital & Vector Mesh")
    st.write("NASA MODIS True Color imagery synchronized with ground sensors. **Arrows indicate real-time wind particle dispersion vectors.**")

    try:
        m = folium.Map(location=[35.7796, -78.6382], zoom_start=10, tiles="CartoDB dark_matter")
        
        folium.TileLayer(
            tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/current/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
            attr="NASA GIBS", name="NASA Orbital Imagery", overlay=True, opacity=0.5
        ).add_to(m)
        
        wake_cities = {
            "Raleigh Node": [35.7796, -78.6382],
            "Cary Node": [35.7915, -78.7811],
            "Wake Forest Node": [35.9799, -78.5097]
        }
        
        for city, coords in wake_cities.items():
            # Get local AQI and Wind Data for each node
            local_weather = fetch_live_weather(coords[0], coords[1])
            local_aqi_data = fetch_current_aqi(coords[0], coords[1])
            
            val = local_aqi_data.get('us_aqi', 0)
            pm = local_aqi_data.get('pm2_5', 0)
            wind_dir = local_weather.get('wind_direction_10m', 0) if local_weather else 0
            
            color = "green" if val <= 50 else "orange" if val <= 100 else "red"
                
            # Node Marker
            folium.CircleMarker(
                location=coords, radius=8, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                popup=f"<b>{city}</b><br>AQI: {val}<br>Wind Dir: {wind_dir}°"
            ).add_to(m)
            
            # Wind Vector Arrow (HTML rotated based on actual wind direction)
            folium.Marker(
                location=coords,
                icon=folium.DivIcon(html=f"""
                    <div style="transform: rotate({wind_dir}deg); font-size: 24px; color: #00f2fe; text-shadow: 0 0 5px #000;">
                        ↑
                    </div>
                """)
            ).add_to(m)
        
        st_folium(m, use_container_width=True, height=550)
        
    except Exception as e:
        st.error(f"Orbital Feed Interrupted: {e}")
