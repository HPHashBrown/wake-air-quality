import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import st_folium

# ============================================
# PAGE CONFIG & HIGH-TECH THEMING
# ============================================
st.set_page_config(page_title="Wake AQI Intelligence", page_icon="🌐", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Glassmorphism, Animations, and Dark/Neon aesthetics
st.markdown("""
    <style>
    /* Animated Global Background */
    .stApp {
        background-color: #0b0f19;
        color: #ffffff;
    }
    
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

    .sub-font {
        font-size: 20px !important; 
        color: #8b9bb4; 
        margin-bottom: 35px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Glassmorphism Metric Cards */
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
    
    .glass-card:hover {
        transform: translateY(-8px);
        border: 1px solid rgba(79, 172, 254, 0.4);
        box-shadow: 0 12px 40px 0 rgba(79, 172, 254, 0.2);
    }
    
    .glass-card h5 {
        color: #8b9bb4 !important;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    
    .glass-card h3 {
        color: #ffffff !important;
        font-size: 38px;
        font-weight: 800;
        margin: 0px 0px 10px 0px;
    }
    
    .glass-card span {
        font-size: 13px;
        color: #4facfe;
    }

    /* Status Badges */
    .badge {
        padding: 6px 12px; 
        border-radius: 20px; 
        font-weight: 800; 
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(0,0,0,0.3);
    }
    .good-badge { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; }
    .mod-badge { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #d97706; }
    .unh-badge { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #dc2626; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# DATA FETCHING (API)
# ============================================
EMAIL = st.secrets.get("EPA_EMAIL", "test@example.com")
API_KEY = st.secrets.get("EPA_API_KEY", "testkey")

@st.cache_data(ttl=3600)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    try:
        res = requests.get(url).json()
        return res.get("current", None)
    except: return None

@st.cache_data(ttl=86400)
def fetch_pm25_epa(year, state_code, county_code):
    url = (f"https://aqs.epa.gov/data/api/dailyData/byCounty?"
           f"email={EMAIL}&key={API_KEY}&param=88101&bdate={year}0101&edate={year}1231"
           f"&state={state_code}&county={county_code}")
    try:
        res = requests.get(url).json()
        if "Data" not in res: return None
        df = pd.DataFrame(res["Data"])
        return df[["date_local", "arithmetic_mean"]] if not df.empty else None
    except: return None

@st.cache_data(ttl=3600)
def fetch_current_aqi(lat=35.7796, lon=-78.6382):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,pm2_5"
    try:
        res = requests.get(url).json()
        return res.get("current", {})
    except: 
        return {}

def fetch_pm25(year):
    return fetch_pm25_epa(year, "37", "183")

def get_aqi_badge(value):
    if value <= 50: return "<div class='badge good-badge'>🟢 Optimal</div>"
    elif value <= 100: return "<div class='badge mod-badge'>🟡 Elevated</div>"
    else: return "<div class='badge unh-badge'>🔴 Hazardous</div>"

# ============================================
# PROCESSING & MODELING
# ============================================
try:
    df_yearly = pd.read_csv("wake_pm25_by_year.csv").sort_values("year")
except:
    df_yearly = pd.DataFrame({"year": [2020, 2021, 2022, 2023], "mean_pm25": [8.5, 9.2, 8.1, 7.9]})

current_year = datetime.now().year
with st.spinner("Initializing Atmospheric Sensors..."):
    df_daily = fetch_pm25(current_year)
    current_aqi_data = fetch_current_aqi()
    live_weather = fetch_live_weather(35.7796, -78.6382)

if df_daily is not None:
    mean_pm25 = df_daily["arithmetic_mean"].mean()
    new_row = pd.DataFrame({"year": [current_year], "mean_pm25": [mean_pm25]})
    df_yearly = df_yearly[df_yearly["year"] != current_year]
    df_yearly = pd.concat([df_yearly, new_row], ignore_index=True).sort_values("year")

X = df_yearly[["year"]]
y = df_yearly["mean_pm25"]
model = LinearRegression().fit(X, y)
future_years = np.arange(current_year + 1, current_year + 11)
future_preds = model.predict(future_years.reshape(-1, 1))
future_df = pd.DataFrame({"year": future_years, "predicted_pm25": future_preds})

current_aqi = current_aqi_data.get('us_aqi', 0)

# ============================================
# UI LAYOUT
# ============================================
st.markdown('<p class="title-gradient">Wake AQI Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Next-Generation Atmospheric PM2.5 Analytics Engine.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Telemetry Overview", 
    "🗃️ Data Archive", 
    "🩺 Bio-Impact", 
    "🧠 Predictive Core", 
    "🛰️ Orbital Feed"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.markdown("### 📡 Live Environmental Context")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{current_aqi}</h3>{get_aqi_badge(current_aqi)}</div>", unsafe_allow_html=True)
    with m2:
        val = f"{live_weather['temperature_2m']}°C" if live_weather else "N/A"
        st.markdown(f"<div class='glass-card'><h5>Thermal State</h5><h3>{val}</h3><span>Raleigh Node</span></div>", unsafe_allow_html=True)
    with m3:
        val = f"{live_weather['relative_humidity_2m']}%" if live_weather else "N/A"
        st.markdown(f"<div class='glass-card'><h5>Saturation</h5><h3>{val}</h3><span>Hourly Feed</span></div>", unsafe_allow_html=True)
    with m4:
        val = f"{live_weather['wind_speed_10m']} km/h" if live_weather else "N/A"
        st.markdown(f"<div class='glass-card'><h5>Wind Velocity</h5><h3>{val}</h3><span>Dispersion Rate</span></div>", unsafe_allow_html=True)

    # Upgraded Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_yearly["year"], y=df_yearly["mean_pm25"], 
        mode="lines+markers", name="Recorded Telemetry", 
        line=dict(color="#00f2fe", width=4),
        marker=dict(size=8, color="#ffffff", line=dict(width=2, color="#00f2fe")),
        fill='tozeroy', fillcolor='rgba(0, 242, 254, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=future_df["year"], y=future_df["predicted_pm25"], 
        mode="lines+markers", name="Algorithmic Forecast", 
        line=dict(color="#f87171", width=4, dash="dot"),
        marker=dict(size=8, color="#ffffff", line=dict(width=2, color="#f87171"))
    ))
    fig.update_layout(
        title="PM2.5 Long-Term Atmospheric Trajectory", 
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: DATA EXPLORER ---
with tab2:
    st.markdown("### 🗃️ Raw Telemetry Archives")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Historical PM2.5 Matrices**")
        st.dataframe(df_yearly, hide_index=True, use_container_width=True)
        st.download_button("📥 Export Historical Feed (CSV)", df_yearly.to_csv(index=False).encode('utf-8'), "historical_pm25.csv")
    with c2:
        st.markdown("**Forecasted Trajectories**")
        st.dataframe(future_df, hide_index=True, use_container_width=True)
        st.download_button("📥 Export Forecast Feed (CSV)", future_df.to_csv(index=False).encode('utf-8'), "forecast_pm25.csv")

# --- TAB 3: HEALTH LITERACY ---
with tab3:
    st.markdown("### 🩺 Biological Impact Protocols")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("#### The Wake County Vector")
        st.write("As a premier hub for biotechnology in North Carolina, Wake County's atmospheric composition is a critical vector for public health. Rapid urban scaling inherently triggers traffic emission spikes, directly inflating PM2.5 concentrations.")
        st.info("System Note: Linear Regression algorithms power these predictions. While highly optimized for longitudinal trends, they cannot foresee sudden acute anomalies such as wildfire drift.")
    with h2:
        st.markdown("#### Medical Glossary")
        with st.expander("🔬 Define: PM2.5"):
            st.write("Microscopic particulate matter (2.5 microns or smaller) capable of bypassing the respiratory barrier and infiltrating the human circulatory system.")
        with st.expander("🛡️ Define: Optimal Levels"):
            st.write("Federal EPA parameters dictate that annual mean concentrations below **12.0 µg/m³** are biologically sustainable for the general population.")

# --- TAB 4: ADVANCED ANALYTICS ---
with tab4:
    st.markdown("### 🧠 Predictive Scenario Core")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Current Threat Level (AQI)**")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=current_aqi, 
            gauge={
                'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': "white"}, 
                'bar': {'color': "#00f2fe"}, 
                'bgcolor': "rgba(255,255,255,0.05)",
                'steps': [
                    {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.3)"}, 
                    {'range': [50, 100], 'color': "rgba(245, 158, 11, 0.3)"}, 
                    {'range': [100, 300], 'color': "rgba(239, 68, 68, 0.3)"}
                ]
            }
        ))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(gauge, use_container_width=True)
    with g2:
        st.markdown("**Emission Mitigation Simulator**")
        st.write("Adjust the theoretical reduction in municipal emissions to view the calculated impact on a 10-year horizon.")
        reduction = st.slider("Simulated Reduction (%)", 0, 50, 0)
        sim_val = future_preds[-1] * (1 - (reduction/100))
        st.metric("Estimated 2034 PM2.5 Concentration", f"{sim_val:.2f} µg/m³", delta=f"-{reduction}% impact")

# --- TAB 5: SATELLITE IMAGERY ---
with tab5:
    st.markdown("### 🛰️ Orbital & Ground Sensor Mesh")
    st.write("Live synchronization of NASA MODIS True Color orbital imagery overlaid with real-time ground sensor arrays.")

    try:
        # High-tech Dark Map
        m = folium.Map(location=[35.7796, -78.6382], zoom_start=10, tiles="CartoDB dark_matter")
        
        # NASA Satellite Layer
        folium.TileLayer(
            tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/current/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
            attr="NASA Global Imagery Browse Services",
            name="NASA Orbital Imagery",
            overlay=True,
            opacity=0.6 # Reduced opacity to blend with the dark theme
        ).add_to(m)
        
        # Sensor Array
        wake_cities = {
            "Raleigh Node": [35.7796, -78.6382],
            "Cary Node": [35.7915, -78.7811],
            "Wake Forest Node": [35.9799, -78.5097],
            "Apex Node": [35.7327, -78.8503],
            "Garner Node": [35.7113, -78.6142]
        }
        
        for city, coords in wake_cities.items():
            live_data = fetch_current_aqi(coords[0], coords[1])
            val = live_data.get('us_aqi', 0)
            pm = live_data.get('pm2_5', 0)
            
            if val <= 50: color = "green"
            elif val <= 100: color = "orange"
            else: color = "red"
                
            folium.Marker(
                location=coords, 
                popup=f"<div style='font-family: monospace; padding: 5px;'><b>{city}</b><br>AQI: {val}<br>PM2.5: {pm} µg/m³</div>",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        st_folium(m, use_container_width=True, height=550)
        
    except Exception as e:
        st.error(f"Orbital Feed Interrupted: {e}")

    st.caption("Data Sources: NASA ESDS & Open-Meteo Telemetry.")
