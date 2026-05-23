import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
from streamlit_autorefresh import st_autorefresh

# ============================================
# INITIALIZATION & STATE
# ============================================
count = st_autorefresh(interval=300000, key="datarefresh")

if 'map_center' not in st.session_state:
    st.session_state.map_center = [35.7796, -78.6382] # Default to Raleigh

if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

OWM_API_KEY = "c76bdd1b10473dad3fd4325a10b954dc"
FIRMS_API_KEY = "5ced48a900256b1fac376db945c3980d" 

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

POLLUTANT_MAP = {
    "PM2.5 (Fine Particulates)": "pm2_5",
    "PM10 (Dust/Coarse)": "pm10",
    "Ozone (O₃)": "ozone",
    "Nitrogen Dioxide (NO₂)": "nitrogen_dioxide",
    "Carbon Monoxide (CO)": "carbon_monoxide",
    "Sulphur Dioxide (SO₂)": "sulphur_dioxide"
}

# ============================================
# API FUNCTIONS
# ============================================

@st.cache_data(ttl=60)
def fetch_global_aqi(lat, lon):
    """The Primary Global Sensor Engine using OpenWeatherMap"""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'list' in data and len(data['list']) > 0:
                main_info = data['list'][0]
                components = main_info.get('components', {})
                # Normalize OWM AQI (1-5 scale) to a visual 0-200 scale
                components['us_aqi'] = main_info.get('main', {}).get('aqi', 0) * 30
                return components
        return {"error": f"Sensor Offline (Code: {response.status_code})"}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600)
def fetch_pollutant_data(lat, lon, pollutant_key):
    """Fallback Local Engine for Raleigh Dashboard"""
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,{pollutant_key}"
    try: 
        return requests.get(url).json().get("current", {})
    except: 
        return {}

@st.cache_data(ttl=3600)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    try:
        return requests.get(url).json().get("current", None)
    except: return None

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
    selected_name = st.selectbox("Pollutant Focus", options=list(POLLUTANT_MAP.keys()))
    selected_key = POLLUTANT_MAP[selected_name]
    
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

# Raleigh Default Logic
current_pollutant_data = fetch_pollutant_data(35.7796, -78.6382, selected_key)
live_weather = fetch_live_weather(35.7796, -78.6382)
current_aqi = current_pollutant_data.get('us_aqi', 0)

# ============================================
# MAIN UI
# ============================================
st.markdown('<p class="title-gradient">Project Wake AQI</p>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Telemetry", "🧠 Prediction", "🩺 Health", "🛰️ Vector Map", "🌌 NASA Space"])

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{current_aqi}</h3></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='glass-card'><h5>Temp</h5><h3>{live_weather['temperature_2m'] if live_weather else 'N/A'}°C</h3></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='glass-card'><h5>Wind</h5><h3>{live_weather['wind_speed_10m'] if live_weather else 'N/A'} km/h</h3></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='glass-card'><h5>Drift</h5><h3>{live_weather['wind_direction_10m'] if live_weather else 'N/A'}°</h3></div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], name="Telemetry", line=dict(color="#00f2fe")))
    fig.add_trace(go.Scatter(x=future_df["year"], y=future_df["predicted_pm25"], name="Forecast", line=dict(color="#f87171", dash="dot")))
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 🔍 Global Sensor Search")
    city_input = st.text_input("Enter City (e.g. Beijing, Moscow, Raleigh)", key="city_search")
    
    if city_input:
        lat, lon, name = get_city_coords(city_input)
        if lat and [lat, lon] != st.session_state.map_center:
            st.session_state.map_center = [lat, lon]
            st.rerun()

    # Map Render
    m = folium.Map(location=st.session_state.map_center, zoom_start=8, tiles="CartoDB dark_matter")
    folium.Marker(st.session_state.map_center, tooltip="Active Hub").add_to(m)
    map_data = st_folium(m, width="100%", height=400, key="map_view")
    
    if map_data and map_data.get('last_clicked'):
        new_coords = [map_data['last_clicked']['lat'], map_data['last_clicked']['lng']]
        if new_coords != st.session_state.map_center:
            st.session_state.map_center = new_coords
            st.rerun()
    
    # INDENTED DATA DISPLAY SECTION
    st.markdown("### 📊 Atmospheric Report")
    curr_lat, curr_lon = st.session_state.map_center
    data = fetch_global_aqi(curr_lat, curr_lon)
    
    if "error" in data:
        st.error(f"⚠️ {data['error']}")
    else:
        st.metric("Global Index Point", f"{data.get('us_aqi', 'N/A')}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("PM2.5", f"{data.get('pm2_5', 'N/A')} µg/m³")
        c2.metric("PM10", f"{data.get('pm10', 'N/A')} µg/m³")
        c3.metric("Ozone", f"{data.get('o3', 'N/A')} µg/m³")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("NO₂", f"{data.get('no2', 'N/A')} µg/m³")
        c5.metric("CO", f"{data.get('co', 'N/A')} µg/m³")
        c6.metric("SO₂", f"{data.get('so2', 'N/A')} µg/m³")

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
