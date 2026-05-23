import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ============================================
# INITIALIZATION & STATE
# ============================================
st_autorefresh(interval=300000, key="datarefresh")

if 'map_center' not in st.session_state:
    st.session_state.map_center = [35.7796, -78.6382] # Default to Raleigh

if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

FIRMS_API_KEY = "5ced48a900256b1fac376db945c3980d" 

# ============================================
# API FUNCTIONS (Restored to stable basics)
# ============================================

@st.cache_data(ttl=3600)
def fetch_pollutant_data(lat, lon):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,pm2_5"
    try: 
        return requests.get(url).json().get("current", {})
    except: 
        return {}

@st.cache_data(ttl=3600)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    try:
        return requests.get(url).json().get("current", {})
    except: 
        return {}

@st.cache_data(ttl=86400)
def get_nasa_climate_data(lat, lon):
    target_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,WS2M&community=RE&longitude={lon}&latitude={lat}&start={target_date}&end={target_date}&format=JSON"
    try:
        response = requests.get(url).json()
        data = response['properties']['parameter']
        solar = list(data['ALLSKY_SFC_SW_DWN'].values())[0]
        wind = list(data['WS2M'].values())[0]
        return {"solar_radiation": solar if solar != -999 else "N/A", "satellite_wind_speed": wind if wind != -999 else "N/A"}
    except:
        return {"solar_radiation": "N/A", "satellite_wind_speed": "N/A"}

@st.cache_data(ttl=3600)
def fetch_wildfire_data():
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_API_KEY}/VIIRS_SNPP_NRT/USA/1"
    try:
        df = pd.read_csv(url)
        if 'lat' in df.columns: df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
        return df[(df['latitude'] >= 34) & (df['latitude'] <= 37) & (df['longitude'] >= -84) & (df['longitude'] <= -75)]
    except: return pd.DataFrame()

@st.cache_data(ttl=86400)
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
# PAGE CONFIG & STYLING
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
    st.markdown("Monitoring Primary Pollutant: **PM2.5**")
    st.markdown("---")
    elapsed = datetime.now() - st.session_state.start_time
    st.write(f"🕒 Node Uptime: **{int(elapsed.total_seconds() // 60)}m**")
    if st.button("🔄 System Reset"):
        st.session_state.start_time = datetime.now()
        st.rerun()

# ============================================
# DATA CORE
# ============================================
df_yearly = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022, 2023], "mean_pm25": [10.2, 9.8, 8.5, 9.2, 8.1, 7.9]})
X = df_yearly[["year"]]
y = df_yearly["mean_pm25"]
model = LinearRegression().fit(X, y)
future_years = np.arange(2024, 2035)
future_preds = model.predict(future_years.reshape(-1, 1))
future_df = pd.DataFrame({"year": future_years, "predicted_pm25": future_preds, "yhat_lower": future_preds - 1.5, "yhat_upper": future_preds + 1.5})

# Fetch Data
curr_lat, curr_lon = st.session_state.map_center
current_data = fetch_pollutant_data(curr_lat, curr_lon)
live_weather = fetch_live_weather(curr_lat, curr_lon)

# ============================================
# MAIN UI
# ============================================
st.markdown('<p class="title-gradient">Project Wake AQI</p>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Telemetry", "🧠 Prediction", "🩺 Health", "🛰️ Vector Map", "🌌 NASA Space"])

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{current_data.get('us_aqi', 'N/A')}</h3></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='glass-card'><h5>PM2.5</h5><h3>{current_data.get('pm2_5', 'N/A')} µg/m³</h3></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='glass-card'><h5>Temp</h5><h3>{live_weather.get('temperature_2m', 'N/A')}°C</h3></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='glass-card'><h5>Wind</h5><h3>{live_weather.get('wind_speed_10m', 'N/A')} km/h</h3></div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], name="Telemetry", line=dict(color="#00f2fe")))
    fig.add_trace(go.Scatter(x=future_df["year"], y=future_df["predicted_pm25"], name="Forecast", line=dict(color="#f87171", dash="dot")))
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🧠 Forecasting Module")
    st.info("Linear Regression model predicting PM2.5 trends over the next decade.")
    st.dataframe(future_df)

with tab3:
    st.markdown("### 🩺 Health Analysis")
    pm25_val = current_data.get('pm2_5', 0)
    # Check to ensure we don't compare a string 'N/A' to an integer
    if pm25_val != 'N/A' and isinstance(pm25_val, (int, float)) and pm25_val > 35:
        st.warning("⚠️ High PM2.5 detected. Sensitive groups should reduce outdoor exertion.")
    else:
        st.success("✅ PM2.5 levels are within acceptable limits.")

with tab4:
    st.markdown("### 🔍 Global Sensor Search")
    city_input = st.text_input("Enter City (e.g. Beijing, Moscow, Raleigh)", key="city_search")
    
    if city_input:
        lat, lon, name = get_city_coords(city_input)
        if lat and [lat, lon] != st.session_state.map_center:
            st.session_state.map_center = [lat, lon]
            st.rerun()

    m = folium.Map(location=st.session_state.map_center, zoom_start=8, tiles="CartoDB dark_matter")
    folium.Marker(st.session_state.map_center, tooltip="Active Hub").add_to(m)
    map_data = st_folium(m, width="100%", height=400, key="map_view")
    
    if map_data and map_data.get('last_clicked'):
        new_coords = [map_data['last_clicked']['lat'], map_data['last_clicked']['lng']]
        if new_coords != st.session_state.map_center:
            st.session_state.map_center = new_coords
            st.rerun()

with tab5:
    st.markdown("### 🌍 NASA Space Intelligence")
    nasa = get_nasa_climate_data(st.session_state.map_center[0], st.session_state.map_center[1])
    col1, col2 = st.columns(2)
    col1.metric("Solar Irradiance", f"{nasa['solar_radiation']} kW/m²")
    col2.metric("Satellite Wind", f"{nasa['satellite_wind_speed']} m/s")
    
    fires = fetch_wildfire_data()
    if not fires.empty:
        st.warning(f"Detected {len(fires)} active heat signatures in regional proximity.")
        st.dataframe(fires[['latitude', 'longitude', 'acq_time']].head(10))
