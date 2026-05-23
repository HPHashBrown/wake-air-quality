import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ============================================
# INITIALIZATION & STATE
# ============================================
st_autorefresh(interval=300000, key="datarefresh")

if 'map_center' not in st.session_state:
    st.session_state.map_center = [35.7796, -78.6382] # Raleigh default

if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

# Mapping internal keys to Open-Meteo API keys
POLLUTANT_MAP = {
    "PM2.5 (Fine Particulates)": "pm2_5",
    "PM10 (Dust/Coarse)": "pm10",
    "Ozone (O₃)": "ozone",
    "Nitrogen Dioxide (NO₂)": "nitrogen_dioxide",
    "Carbon Monoxide (CO)": "carbon_monoxide",
    "Sulphur Dioxide (SO₂)": "sulphur_dioxide"
}

# ============================================
# BULLETPROOF DATA ENGINE
# ============================================

@st.cache_data(ttl=300)
def get_atmospheric_data(lat, lon):
    """Fetches all AQI and Pollutant data from Open-Meteo (No API Key Required)"""
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("current", {})
        return {"error": f"Station Offline ({response.status_code})"}
    except:
        return {"error": "Connection Timeout"}

@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    try:
        return requests.get(url).json().get("current", {})
    except:
        return {}

@st.cache_data(ttl=3600)
def get_city_coords(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    try:
        res = requests.get(url).json()
        if "results" in res:
            d = res["results"][0]
            return d["latitude"], d["longitude"], d["name"]
        return None, None, None
    except: return None, None, None

# ============================================
# UI CONFIG & STYLE
# ============================================
st.set_page_config(page_title="Project Wake AQI", page_icon="🌐", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #ffffff; }
    .title-gradient {
        background: linear-gradient(-45deg, #00f2fe, #4facfe, #00f2fe, #4facfe);
        background-size: 300% 300%;
        animation: gradient-shift 8s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 54px !important; font-weight: 900; text-transform: uppercase;
    }
    @keyframes gradient-shift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .glass-card {
        background: rgba(16, 25, 43, 0.6); backdrop-filter: blur(12px); border-radius: 16px; 
        border: 1px solid rgba(255, 255, 255, 0.08); padding: 25px; text-align: center; margin-bottom: 15px;
    }
    .glass-card h5 { color: #8b9bb4 !important; font-size: 14px; text-transform: uppercase; }
    .glass-card h3 { font-size: 38px; font-weight: 800; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 🧪 Atmospheric Lab")
    selected_name = st.selectbox("Pollutant Focus", options=list(POLLUTANT_MAP.keys()))
    selected_key = POLLUTANT_MAP[selected_name]
    
    st.markdown("---")
    elapsed = datetime.now() - st.session_state.start_time
    st.write(f"🕒 Node Uptime: **{int(elapsed.total_seconds() // 60)}m**")

# ============================================
# MAIN INTERFACE
# ============================================
st.markdown('<p class="title-gradient">Project Wake AQI</p>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Telemetry", "🧠 Prediction", "🩺 Health", "🛰️ Vector Map"])

# Fetch shared data for the current map center
current_aqi_data = get_atmospheric_data(st.session_state.map_center[0], st.session_state.map_center[1])
weather_data = fetch_live_weather(st.session_state.map_center[0], st.session_state.map_center[1])

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{current_aqi_data.get('us_aqi', 'N/A')}</h3></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='glass-card'><h5>Temp</h5><h3>{weather_data.get('temperature_2m', 'N/A')}°C</h3></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='glass-card'><h5>Wind</h5><h3>{weather_data.get('wind_speed_10m', 'N/A')} km/h</h3></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='glass-card'><h5>Humidity</h5><h3>{weather_data.get('relative_humidity_2m', 'N/A')}%</h3></div>", unsafe_allow_html=True)

    # Placeholder for the Plotly Chart (Using your original logic)
    df_yearly = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022, 2023], "mean_pm25": [10.2, 9.8, 8.5, 9.2, 8.1, 7.9]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], name="Telemetry", line=dict(color="#00f2fe")))
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 🔍 Global Sensor Search")
    city_col1, city_col2 = st.columns([3, 1])
    with city_col1:
        city_input = st.text_input("Enter City", placeholder="e.g. Tokyo, Raleigh, Berlin", key="city_search")
    
    if city_input:
        lat, lon, name = get_city_coords(city_input)
        if lat:
            st.session_state.map_center = [lat, lon]
            # No st.rerun needed here as it will refresh on next interaction

    # Map Render
    m = folium.Map(location=st.session_state.map_center, zoom_start=10, tiles="CartoDB dark_matter")
    folium.Marker(st.session_state.map_center, tooltip="Current Focus").add_to(m)
    map_data = st_folium(m, width="100%", height=400, key="map_view")
    
    if map_data and map_data.get('last_clicked'):
        new_coords = [map_data['last_clicked']['lat'], map_data['last_clicked']['lng']]
        if new_coords != st.session_state.map_center:
            st.session_state.map_center = new_coords
            st.rerun()
    
    # Atmospheric Report logic moved fully inside Tab 4
    st.markdown("### 📊 Atmospheric Report")
    
    if "error" in current_aqi_data:
        st.error(f"⚠️ {current_aqi_data['error']}")
    else:
        c1, c2, c3 = st.columns(3)
        # Using standardized Open-Meteo keys
        c1.metric("PM2.5", f"{current_aqi_data.get('pm2_5', 'N/A')} µg/m³")
        c2.metric("PM10", f"{current_aqi_data.get('pm10', 'N/A')} µg/m³")
        c3.metric("Ozone", f"{current_aqi_data.get('ozone', 'N/A')} µg/m³")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("NO₂", f"{current_aqi_data.get('nitrogen_dioxide', 'N/A')} µg/m³")
        c5.metric("CO", f"{current_aqi_data.get('carbon_monoxide', 'N/A')} µg/m³")
        c6.metric("SO₂", f"{current_aqi_data.get('sulphur_dioxide', 'N/A')} µg/m³")
