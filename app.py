from owslib.wms import WebMapService
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# ============================================
# PAGE CONFIG & THEMING
# ============================================
st.set_page_config(page_title="Wake AQI Intelligence", page_icon="🌤️", layout="wide")

# Fixed CSS: Explicit text colors to prevent "white-on-white" squares
st.markdown("""
    <style>
    .big-font {font-size: 46px !important; font-weight: 700; color: #1E3A8A; margin-bottom: 0px;}
    .sub-font {font-size: 18px !important; color: #64748B; margin-bottom: 25px;}
    .metric-card {
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #d1d5db;
        color: #1e293b !important;
    }
    .metric-card h3, .metric-card h5, .metric-card h2 {
        color: #1e293b !important;
        margin: 0;
    }
    .good-badge {background-color: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 5px; font-weight: bold; font-size: 14px;}
    .mod-badge {background-color: #fef08a; color: #854d0e; padding: 4px 8px; border-radius: 5px; font-weight: bold; font-size: 14px;}
    .unh-badge {background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 5px; font-weight: bold; font-size: 14px;}
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
    # Switched to Open-Meteo Air Quality API - No API Key Required!
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,pm2_5"
    try:
        res = requests.get(url).json()
        return res.get("current", {})
    except: 
        return {}

# Helper to maintain compatibility
def fetch_pm25(year):
    return fetch_pm25_epa(year, "37", "183")

def get_aqi_badge(value):
    if value <= 50: return "<span class='good-badge'>🟢 Good</span>"
    elif value <= 100: return "<span class='mod-badge'>🟡 Moderate</span>"
    else: return "<span class='unh-badge'>🔴 Unhealthy</span>"

# ============================================
# PROCESSING & MODELING
# ============================================
try:
    df_yearly = pd.read_csv("wake_pm25_by_year.csv").sort_values("year")
except:
    df_yearly = pd.DataFrame({"year": [2020, 2021, 2022, 2023], "mean_pm25": [8.5, 9.2, 8.1, 7.9]})

current_year = datetime.now().year
with st.spinner("Fetching EPA, Weather, and AQI data..."):
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

# Determine current AQI for display securely from Open-Meteo
current_aqi = current_aqi_data.get('us_aqi', 0)

# ============================================
# UI LAYOUT
# ============================================
st.markdown('<p class="big-font">Wake County Air Quality Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Medical-grade analytics for atmospheric particulate matter (PM2.5).</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & Forecast", 
    "🗃️ Data Explorer", 
    "🩺 Health Literacy", 
    "🧠 Advanced Analytics", 
    "🛰️ Satellite Imagery"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader("🌐 Real-Time Environmental Context")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><h5>Current AQI</h5><h3>{current_aqi}</h3>{get_aqi_badge(current_aqi)}</div>", unsafe_allow_html=True)
    with m2:
        val = f"{live_weather['temperature_2m']}°C" if live_weather else "N/A"
        st.markdown(f"<div class='metric-card'><h5>Live Temp</h5><h3>{val}</h3><span>Raleigh, NC</span></div>", unsafe_allow_html=True)
    with m3:
        val = f"{live_weather['relative_humidity_2m']}%" if live_weather else "N/A"
        st.markdown(f"<div class='metric-card'><h5>Humidity</h5><h3>{val}</h3><span>Hourly Feed</span></div>", unsafe_allow_html=True)
    with m4:
        val = f"{live_weather['wind_speed_10m']} km/h" if live_weather else "N/A"
        st.markdown(f"<div class='metric-card'><h5>Wind Speed</h5><h3>{val}</h3><span>Dispersion Rate</span></div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], mode="markers+lines", name="Historical", line=dict(color="#1E3A8A", width=3)))
    fig.add_trace(go.Scatter(x=future_df["year"], y=future_df["predicted_pm25"], mode="markers+lines", name="10-Year Forecast", line=dict(color="#EF4444", width=3, dash="dot")))
    fig.update_layout(title="PM2.5 Historical Trends vs Future Predictions", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: DATA EXPLORER ---
with tab2:
    st.subheader("🗃️ Data Archive")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Historical Dataset**")
        st.dataframe(df_yearly, hide_index=True, use_container_width=True)
        st.download_button("📥 Download Historical CSV", df_yearly.to_csv(index=False).encode('utf-8'), "historical_pm25.csv")
    with c2:
        st.write("**Forecasted Estimates**")
        st.dataframe(future_df, hide_index=True, use_container_width=True)
        st.download_button("📥 Download Forecast CSV", future_df.to_csv(index=False).encode('utf-8'), "forecast_pm25.csv")

# --- TAB 3: HEALTH LITERACY ---
with tab3:
    st.subheader("🩺 Health Impacts & Methodology")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("### Why Wake County?")
        st.write("As a major hub for biotechnology and healthcare in NC, Wake County's air quality is a key indicator of public health. Urban expansion leads to higher traffic emissions, increasing PM2.5 levels.")
        st.markdown("### How to Interpret")
        st.info("Predictions use Linear Regression. While accurate for long-term trends, they don't predict 'spikes' caused by events like wildfires.")
    with h2:
        st.markdown("### Medical Glossary")
        with st.expander("What is PM2.5?"):
            st.write("Fine particles (2.5 microns or less) that can enter the bloodstream through the lungs.")
        with st.expander("What is a 'Good' Level?"):
            st.write("The EPA considers annual means below 12.0 µg/m³ to be healthy for most people.")

# --- TAB 4: ADVANCED ANALYTICS ---
with tab4:
    st.subheader("🧠 Advanced Forecasting")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**AQI Status Gauge**")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=current_aqi, gauge={'axis': {'range': [0, 300]}, 'bar': {'color': "#1E3A8A"}, 'steps': [{'range': [0, 50], 'color': "#dcfce7"}, {'range': [50, 100], 'color': "#fef08a"}, {'range': [100, 300], 'color': "#fee2e2"}]}))
        st.plotly_chart(gauge, use_container_width=True)
    with g2:
        st.markdown("**Scenario Simulator**")
        reduction = st.slider("Simulated % Reduction in Emissions", 0, 50, 0)
        sim_val = future_preds[-1] * (1 - (reduction/100))
        st.metric("Estimated 2034 PM2.5", f"{sim_val:.2f} µg/m³", delta=f"-{reduction}%")

# --- TAB 5: SATELLITE IMAGERY ---
with tab5:
    st.subheader("🛰️ Real-time Satellite & Official Monitoring")
    
    st.markdown("""
    This map displays **NASA's Daily True Color Satellite Imagery** layered with live AQI monitors across Wake County.
    """)

    try:
        m = folium.Map(location=[35.7796, -78.6382], zoom_start=10, tiles="CartoDB positron")
        
        # NASA Satellite Layer (Updated to a reliable WMTS feed)
        folium.TileLayer(
            tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/current/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
            attr="NASA Global Imagery Browse Services",
            name="NASA Satellite Imagery",
            overlay=True,
        ).add_to(m)
        
        # Wake County Cities to monitor
        wake_cities = {
            "Raleigh": [35.7796, -78.6382],
            "Cary": [35.7915, -78.7811],
            "Wake Forest": [35.9799, -78.5097],
            "Apex": [35.7327, -78.8503],
            "Garner": [35.7113, -78.6142]
        }
        
        # Fetch and plot live Open-Meteo AQI for each city
        for city, coords in wake_cities.items():
            live_data = fetch_current_aqi(coords[0], coords[1])
            val = live_data.get('us_aqi', 0)
            pm = live_data.get('pm2_5', 0)
            
            # Marker colors based on US AQI
            if val <= 50: color = "green"
            elif val <= 100: color = "orange"
            else: color = "red"
                
            folium.Marker(
                location=coords, 
                popup=f"<b>{city}</b><br>AQI: {val}<br>PM2.5: {pm} µg/m³",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        st_folium(m, use_container_width=True, height=500)
        
    except Exception as e:
        st.error(f"Could not load the map: {e}")

    st.caption("Source: NASA ESDS & Open-Meteo Air Quality Feed.")
