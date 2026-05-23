import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ============================================
# INITIALIZATION
# ============================================
st_autorefresh(interval=300000, key="datarefresh")

if 'map_center' not in st.session_state:
    st.session_state.map_center = [35.7796, -78.6382] 

# ============================================
# DIAGNOSTIC DATA ENGINE (No Cache, Raw Output)
# ============================================
def get_pm25_diagnostic(lat, lon):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,us_aqi"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # This returns the 'current' dictionary OR an empty dict with the error
        if response.status_code == 200 and "current" in data:
            return data["current"], data # Return both the clean dict and the raw JSON
        else:
            return {"error": f"API Error: {response.status_code}"}, data
    except Exception as e:
        return {"error": str(e)}, {}

# ============================================
# UI CONFIG
# ============================================
st.set_page_config(page_title="Project Wake AQI", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #ffffff; }
    .glass-card {
        background: rgba(16, 25, 43, 0.6); backdrop-filter: blur(12px); border-radius: 12px; 
        border: 1px solid rgba(255, 255, 255, 0.1); padding: 20px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Project Wake: PM2.5 Monitor")

# Fetch data and the raw "Debug" dump
aqi_clean, raw_dump = get_pm25_diagnostic(st.session_state.map_center[0], st.session_state.map_center[1])

# ============================================
# THE DASHBOARD
# ============================================
t1, t2 = st.tabs(["📊 Live Telemetry", "🛰️ Sensor Map"])

with t1:
    # If there is an error in the data, show it clearly
    if "error" in aqi_clean:
        st.error(f"Diagnostic Alert: {aqi_clean['error']}")
    
    col1, col2 = st.columns(2)
    
    # We use .get() but provide 'No Data' instead of N/A to see if it changes
    pm_val = aqi_clean.get('pm2_5', 'No Data')
    aqi_val = aqi_clean.get('us_aqi', 'No Data')
    
    col1.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{aqi_val}</h3></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='glass-card'><h5>PM2.5</h5><h3>{pm_val} µg/m³</h3></div>", unsafe_allow_html=True)

    # --- DEBUG SECTION ---
    with st.expander("🛠️ Developer Diagnostic (See Raw API Response)"):
        st.write("Target Coordinates:", st.session_state.map_center)
        st.write("Raw JSON from Server:", raw_dump)

with t2:
    st.info("Click the map to update the telemetry for that specific coordinate.")
    m = folium.Map(location=st.session_state.map_center, zoom_start=10, tiles="CartoDB dark_matter")
    folium.Marker(st.session_state.map_center).add_to(m)
    map_data = st_folium(m, width="100%", height=400)
    
    if map_data and map_data.get('last_clicked'):
        clicked = [map_data['last_clicked']['lat'], map_data['last_clicked']['lng']]
        if clicked != st.session_state.map_center:
            st.session_state.map_center = clicked
            st.rerun()
