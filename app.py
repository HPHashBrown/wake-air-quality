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

# ============================================
# DATA ENGINE (PM2.5 Focus)
# ============================================

@st.cache_data(ttl=300)
def get_pm25_data(lat, lon):
    """Fetches PM2.5 and US AQI from Open-Meteo"""
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,us_aqi"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("current", {})
        return {"error": "Station Offline"}
    except:
        return {"error": "Connection Timeout"}

@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
    try:
        return requests.get(url).json().get("current", {})
    except:
        return {}

@st.cache_data(ttl=86400)
def get_city_coords(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    try:
        res = requests.get(url).json()
        if "results" in res:
            d = res["results"][0]
            return d["latitude"], d["longitude"]
        return None, None
    except: return None, None

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
# MAIN INTERFACE
# ============================================
st.markdown('<p class="title-gradient">Project Wake AQI</p>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Telemetry", "🧠 Prediction", "🛰️ Vector Map"])

# Fetch shared data for the current map center
aqi_data = get_pm25_data(st.session_state.map_center[0], st.session_state.map_center[1])
weather_data = fetch_live_weather(st.session_state.map_center[0], st.session_state.map_center[1])

with tab1:
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{aqi_data.get('us_aqi', 'N/A')}</h3></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='glass-card'><h5>PM2.5</h5><h3>{aqi_data.get('pm2_5', 'N/A')} µg/m³</h3></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='glass-card'><h5>Temp</h5><h3>{weather_data.get('temperature_2m', 'N/A')}°C</h3></div>", unsafe_allow_html=True)

    # Simple historical placeholder
    df_yearly = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022, 2023], "mean_pm25": [10.2, 9.8, 8.5, 9.2, 8.1, 7.9]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], name="PM2.5 Telemetry", line=dict(color="#00f2fe")))
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 🔍 Regional Analysis")
    city_input = st.text_input("Search City", placeholder="e.g. Raleigh, Tokyo", key="city_search")
    
    if city_input:
        lat, lon = get_city_coords(city_input)
        if lat:
            st.session_state.map_center = [lat, lon]

    # Map Render
    m = folium.Map(location=st.session_state.map_center, zoom_start=10, tiles="CartoDB dark_matter")
    folium.Marker(st.session_state.map_center, tooltip="Current Location").add_to(m)
    map_data = st_folium(m, width="100%", height=400, key="map_view")
    
    if map_data and map_data.get('last_clicked'):
        new_coords = [map_data['last_clicked']['lat'], map_data['last_clicked']['lng']]
        if new_coords != st.session_state.map_center:
            st.session_state.map_center = new_coords
            st.rerun()
    
    # Atmospheric Report simplified to just PM2.5
    st.markdown("### 📊 Local Concentration")
    if "error" in aqi_data:
        st.error(f"⚠️ {aqi_data['error']}")
    else:
        st.metric("Current PM2.5 Concentration", f"{aqi_data.get('pm2_5', 'N/A')} µg/m³")
