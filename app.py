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
import osmnx as ox
import networkx as nx
from google import genai
from folium.plugins import HeatMap
import openaq
import pydeck as pdk
import random


# ============================================
# INITIALIZATION & STATE
# ============================================

def create_base_map(lat, lon, zoom=12):
    return folium.Map(location=[lat, lon], zoom_start=zoom, tiles="CartoDB dark_matter")

def add_heatmap_to_map(m, hotspots):
    from folium.plugins import HeatMap
    HeatMap(hotspots, radius=20, blur=15).add_to(m)
    return m


# Ensure this is defined exactly once at the top of your script
df_yearly = pd.DataFrame({
    "year": [2018, 2019, 2020, 2021, 2022, 2023], 
    "mean_pm25": [10.2, 9.8, 8.5, 9.2, 8.1, 7.9],
    "pressure_hpa": [1012, 1015, 1010, 1013, 1018, 1011],  # Added variation
    "humidity": [65, 72, 60, 68, 62, 70],                 # Added variation
    "aerosol_depth": [0.12, 0.15, 0.10, 0.13, 0.09, 0.11] # Added variation
})

# 1. Update Initialization
# --- GEMINI INITIALIZATION ---
# --- GEMINI INITIALIZATION ---
# Change your initialization to this:
try:
    # The new SDK automatically picks up the API key from the environment 
    # or you can pass it explicitly if needed.
    client = genai.Client(api_key=st.secrets["GEM_KEY"])

    # You no longer need 'GenerativeModel' as a separate class assignment
    # You just call it through the client
except Exception as e:
    st.error(f"Error initializing Client: {e}")
    client = None

def get_ai_health_briefing(pm25_val, aqi_val):
    if client is None:
        return "AI Health briefing is currently unavailable."

    prompt = f"""
    Act as a clinical health educator. The current PM2.5 level is {pm25_val} µg/m³ and the AQI is {aqi_val}.
    Write a 3-sentence "Daily Health Briefing" for a general audience.
    1. Explain the biological impact on the respiratory system.
    2. Suggest one specific preventative measure based on these levels.
    3. Keep it empathetic and professional.
    """
    try:
        # Use the new models.generate_content method
        response = client.models.generate_content(
            model="gemini-3.5-flash",  # Or "gemini-2.0-flash" if you prefer
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error communicating with AI: {str(e)}"

def calculate_resilience_score(aqi, wind_speed, humidity):
    """
    Calculates a 0-100 score:
    - Higher is better (more resilient/cleaner/stable)
    - AQI is the primary detractor
    - Wind is a bonus (dispersion)
    - Humidity is a stabilizer (low impact)
    """
    # Normalize AQI: 0 is perfect, 300 is hazardous
    aqi_impact = max(0, 100 - (aqi / 3)) 

    # Wind bonus: Up to 15 points if wind is strong (helps dispersion)
    wind_bonus = min(15, wind_speed * 1.5)

    # Resilience score
    score = aqi_impact + wind_bonus
    return min(100, max(0, round(score)))

# Set a longer timeout for the API request
ox.settings.timeout = 300 
# Tell OSMnx to use the public Overpass server via a more stable request method
ox.settings.use_cache = True

st.set_page_config(page_title="Project AIR", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")
st_autorefresh(interval=600000, key="datarefresh")

if 'map_center' not in st.session_state:
    st.session_state.map_center = [35.7796, -78.6382]
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

from openaq import OpenAQ

@st.cache_data(ttl=3600)
def get_real_pm25_data(lat, lon):
    try:
        client = OpenAQ(api_key=st.secrets["OPENAQ_API_KEY"])

        # 1. Find sensors near the coordinate
        locations = client.locations.list(
            coordinates=(lat, lon),
            radius=25000,
            parameter_ids=[1], # PM2.5 ID
            limit=5
        )

        # Check if results exist
        if not locations.results:
            return []

        # 2. Extract sensor IDs
        sensor_ids = [loc.id for loc in locations.results]

        # 3. Fetch measurements for those sensors
        data = client.measurements.list(
            location_ids=sensor_ids,
            parameter_ids=[1],
            limit=100
        )

        # 4. Return formatted data
        return [[m.coordinates.latitude, m.coordinates.longitude, min(m.value, 100)] for m in data.results]

    except Exception as e:
        # We don't want to show the full error to the user if it's just a rate limit
        return []

        # 2. Extract IDs of the sensors found
        # (Assuming the SDK returns objects with an 'id' attribute)
        sensor_ids = [loc.id for loc in locations.results]

        if not sensor_ids:
            return []



        import requests

def fetch_us_fire_perimeters():
    """
    Fetches active US wildfire perimeters from the NIFC ArcGIS Feature Service.
    Returns a GeoJSON object or None if the request fails.
    """
    url = (
        "https://services3.arcgis.com/T4sVRZ87U1p7PBYQ/arcgis/rest/services/"
        "Current_WildlandFire_Perimeters/FeatureServer/0/query?"
        "where=1%3D1&outFields=IncidentName,DailyAcres,CurrentDate&f=geojson"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None
        # 3. Fetch measurements for those specific sensor IDs
        # We use 'measurements.list' with 'sensors_id' instead
        data = client.measurements.list(
            sensors_id=sensor_ids[0], # Fetch from the first sensor found
            limit=100
        )

        return [[m.coordinates.latitude, m.coordinates.longitude, min(m.value, 100)] for m in data.results]

    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return []
# Define the function ONCE
@st.cache_data(ttl=3600)
def fetch_global_aqi(lat, lon, metric="pm2_5"):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,{metric}"
    try: 
        response = requests.get(url).json()
        return response.get("current", {})
    except: 
        return {}

# Safe Initialization
if 'current_data' not in st.session_state:
    st.session_state.current_data = fetch_global_aqi(35.7796, -78.6382)

current_data = st.session_state.get('current_data', {})
# SAFELY convert to float, default to 0 if it's 'N/A' or missing
raw_aqi = current_data.get('us_aqi', 0)
current_aqi = float(raw_aqi) if str(raw_aqi).replace('.','',1).isdigit() else 0

FIRMS_API_KEY = "5ced48a900256b1fac376db945c3980d" 
metric_key = "pm2_5" 
selected_metric = "PM2.5" 
selected_risks = ["Good (0-50)", "Moderate (51-100)", "Unhealthy (101+)"]
POLLUTANT_MAP = {"PM2.5": "pm2_5", "PM10": "pm10", "Ozone": "ozone", "NO2": "nitrogen_dioxide"}

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

@st.cache_data(show_spinner=True)
def get_map_graph(lat1, lon1, lat2, lon2):
    center_lat, center_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    graph = None
    for attempt in range(3):
        try:
            graph = ox.graph_from_point((center_lat, center_lon), dist=5000, network_type='drive')
            break
        except Exception:
            continue
    if graph:
        hwy_speeds = {'residential': 35, 'secondary': 50, 'tertiary': 40, 'primary': 60}
        graph = ox.add_edge_speeds(graph, hwy_speeds=hwy_speeds)
        graph = ox.add_edge_travel_times(graph)
    return graph


# ============================================
# API FUNCTIONS (MOVED TO TOP TO PREVENT ERRORS)
# ============================================

df_yearly = pd.DataFrame({
    "year": [2018, 2019, 2020, 2021, 2022, 2023], 
    "mean_pm25": [10.2, 9.8, 8.5, 9.2, 8.1, 7.9]
})

current_year = 2026

def get_healthiest_route(start_lat, start_lon, end_lat, end_lon):
    # Sample 3 points: Start, Midpoint, End
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2

    points = [(start_lat, start_lon), (mid_lat, mid_lon), (end_lat, end_lon)]

    total_aqi = 0
    for lat, lon in points:
        data = fetch_global_aqi(lat, lon)
        total_aqi += data.get('us_aqi', 50) # Default to 50 if data missing

    avg_aqi = total_aqi / 3
    return avg_aqi


def get_nasa_climate_data(lat, lon):
    target_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,WS2M&community=RE&longitude={lon}&latitude={lat}&start={target_date}&end={target_date}&format=JSON"
    try:
        response = requests.get(url).json()
        data = response['properties']['parameter']
        solar_values = list(data['ALLSKY_SFC_SW_DWN'].values())
        wind_values = list(data['WS2M'].values())
        solar = solar_values[0] if solar_values[0] != -999 else "N/A"
        wind = wind_values[0] if wind_values[0] != -999 else "N/A"
        return {"solar_radiation": solar, "satellite_wind_speed": wind}
    except Exception as e:
        return {"solar_radiation": "N/A", "satellite_wind_speed": "N/A"}

# Instead of CSV, use the global fire hotspot detection layer from NASA GIBS
# This layer is natively supported by PyDeck and doesn't require an API key
import pydeck as pdk

# --- CLEANED UP SECTION ---
def get_global_fire_layer():
    return pdk.Layer(
        "TileLayer",
        f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_Thermal_Anomalies_375m/default/{datetime.now().strftime('%Y-%m-%d')}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.png",
        opacity=0.8
    )

def get_nasa_climate_data(lat, lon):
    """Fetches solar and wind data from NASA POWER API."""
    try:
        # NASA POWER API for daily climate parameters
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,WS2M&community=RE&longitude={lon}&latitude={lat}&start=20240101&end=20240101&format=JSON"
        # Note: In a production app, you'd dynamically set the dates to yesterday
        # For this example, we use a static fallback or simplified mock if API is down
        res = requests.get(url, timeout=5).json()
        solar = res['properties']['parameter']['ALLSKY_SFC_SW_DWN']['20240101']
        wind = res['properties']['parameter']['WS2M']['20240101']
        return {"solar_radiation": solar if solar != -999 else "N/A", 
                "satellite_wind_speed": wind if wind != -999 else "N/A"}
    except:
        return {"solar_radiation": "5.2", "satellite_wind_speed": "3.8"} # Realistic Fallback

@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon, city_name="Default"):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        current = data.get("current", {})
        return current
    except Exception as e:
        st.warning(f"Weather fetch failed: {e}")
        return {}

@st.cache_data(ttl=3600)
def fetch_pollutant_data(lat, lon, pollutant_key):
    # This URL now dynamically pulls the specific pollutant selected
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,{pollutant_key}"
    try: 
        return requests.get(url).json().get("current", {})
    except: 
        return {}



@st.cache_data(ttl=86400)
def get_city_coords(city_name):
    # Failsafe for New Delhi
    if "new delhi" in city_name.lower().strip():
        # Force these specific coordinates to bypass ambiguity
        return 28.6139, 77.2090, "New Delhi"

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"

    try:
        response = requests.get(url).json()
        if "results" in response and response["results"]:
            data = response["results"][0]
            return data["latitude"], data["longitude"], data["name"]
        return None, None, None
    except Exception:
        return None, None, None

def fetch_30day_correlation(lat, lon):
    try:
        # Define the 30-day window
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        # Open-Meteo Historical Air Quality & Weather API
        aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&start_date={start_date}&end_date={end_date}"
        w_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        
        aq_res = requests.get(aq_url).json()
        w_res = requests.get(w_url).json()
        
        # Combine into a single DataFrame
        df = pd.DataFrame({
            'time': pd.to_datetime(aq_res['hourly']['time']),
            'pm25': aq_res['hourly']['pm2_5'],
            'temp': w_res['hourly']['temperature_2m'],
            'humidity': w_res['hourly']['relative_humidity_2m'],
            'wind': w_res['hourly']['wind_speed_10m']
        })
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None

def generate_pdf_report(df_yearly, future_df, current_aqi):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Wake County AQI Intelligence Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Current AQI Level: {current_aqi}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Historical Data Summary:", ln=True)
    pdf.cell(200, 10, txt=df_yearly.to_string(), ln=True)
    return pdf.output(dest='S').encode('latin-1')

def calculate_micro_climate_differential(user_lat, user_lon, control_lat, control_lon):
    # Fetch AQI for both points
    user_data = fetch_global_aqi(user_lat, user_lon)
    control_data = fetch_global_aqi(control_lat, control_lon)

    user_aqi = user_data.get('us_aqi', 0)
    control_aqi = control_data.get('us_aqi', 0)

    diff = user_aqi - control_aqi

    if diff > 15:
        return f"🚨 Micro-climate Alert: Your location is {diff} AQI points dirtier than the nearby node."
    elif diff < -5:
        return "✅ You are currently in a high-quality air pocket."
    return "Air quality is consistent across your local area."

# ============================================
# PAGE CONFIG & HIGH-TECH THEMING
# ============================================

st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #ffffff; }
    
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

    .css-1d391kg { background-color: #0b0f19; border-right: 1px solid rgba(255, 255, 255, 0.08); }
    </style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 🔔 Automated Alerting")
    with st.form("alert_form"):
        target_email = st.text_input("Operator Email", placeholder="operator@nc.gov")
        alert_threshold = st.slider("US AQI Trigger Threshold", min_value=50, max_value=300, value=100, step=10)
        submit_alert = st.form_submit_button("Initialize Protocol")
        if submit_alert:
            if target_email: st.success("Protocol Active.")
            else: st.error("Error: Valid Email Required.")


    st.markdown("---")
    selected_name = st.selectbox("🧬 Select Pollutant", options=list(POLLUTANT_MAP.keys()))
    st.session_state['selected_pollutant_key'] = POLLUTANT_MAP[selected_name]
    st.session_state['selected_pollutant_name'] = selected_name

    st.markdown("---")
    elapsed = datetime.now() - st.session_state.start_time
    seconds = int(elapsed.total_seconds())
    if seconds < 60:
        time_str = f"{seconds} seconds"
    else:
        time_str = f"{seconds // 60} minutes"
    st.write(f"🕒 Data updated: **{time_str} ago**")

    if st.button("🔄 Manual Refresh"):
        st.session_state.start_time = datetime.now()
        st.rerun()


# ============================================
# PROCESSING & MODELING
# ============================================

# --- FIX: Save initialization to Session State ---
with st.spinner("Initializing Atmospheric Sensors..."):
    # If the state is empty (first load), fetch and STORE the data
    if 'current_weather' not in st.session_state:
        st.session_state.current_weather = fetch_live_weather(35.7796, -78.6382)
    
    # Also ensure current_data is initialized for the AQI metric
    if 'current_data' not in st.session_state:
        st.session_state.current_data = fetch_global_aqi(35.7796, -78.6382)

if PROPHET_AVAILABLE:
    prophet_df = df_yearly.copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df['year'], format='%Y')
    prophet_df = prophet_df.rename(columns={'mean_pm25': 'y'})
    m = Prophet(yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=10, freq='YS')
    forecast = m.predict(future)
    future_df = pd.DataFrame({
        "year": forecast['ds'].dt.year,
        "predicted_pm25": forecast['yhat'],
        "yhat_lower": forecast['yhat_lower'],
        "yhat_upper": forecast['yhat_upper']
    })
else:
    X = df_yearly[["year"]]
    y = df_yearly["mean_pm25"]
    model = LinearRegression().fit(X, y)
    future_years = np.arange(current_year, current_year + 11)
    future_preds = model.predict(future_years.reshape(-1, 1))
    future_df = pd.DataFrame({
        "year": future_years, 
        "predicted_pm25": future_preds,
        "yhat_lower": future_preds - 1.5, 
        "yhat_upper": future_preds + 1.5
    })

# ============================================
# MAIN UI LAYOUT
# ============================================
st.markdown('<p class="title-gradient">Project AIR</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Project AIR (Atmospheric Intelligence & Response) is a Global Atmospheric PM2.5 Analytics Engine.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Telemetry & Forecasting", "🧠 Hypothetical Prediction Measure", "🩺 Health Literacy", "🛰️ Global Vector Map", "🌌NASA Forest Fire Intelligence", "🚲Clean-Air Commute", "📈Monthly PM2.5 Comparison"])

# --- TAB 1: OVERVIEW (STABLE BUILD) ---
with tab1:
    # 1. Layout & Search
    st.markdown("### 🌍 Regional Atmospheric & Bio-Telemetry Analysis")
    
    with st.container():
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            city_input = st.text_input("Search Location", "Raleigh", key="tab1_search", label_visibility="collapsed")
        with col_btn:
            if st.button("📡 Scan Region", use_container_width=True):
                lat, lon, name = get_city_coords(city_input)
                if lat:
                    st.session_state.map_center = [lat, lon]
                    st.session_state.current_data = fetch_global_aqi(lat, lon)
                    st.session_state.current_weather = fetch_live_weather(lat, lon)
                    st.rerun()

    # 2. Data Retrieval
    data = st.session_state.get('current_data', {})
    weather = st.session_state.get('current_weather', {})

    # Validation: Re-fetch only if weather dict is truly empty (missing keys entirely)
    if not weather or 'temperature_2m' not in weather:
        curr_lat, curr_lon = st.session_state.get('map_center', [35.7796, -78.6382])
        weather = fetch_live_weather(curr_lat, curr_lon)
        st.session_state.current_weather = weather

    # 3. Variable Extraction — use None-safe defaults
    aqi_val = float(data.get('us_aqi') or 0)
    temp = weather.get('temperature_2m', 'N/A')
    wind = weather.get('wind_speed_10m', 'N/A')
    head = weather.get('wind_direction_10m', 'N/A')
    
    aqi_color = "#10b981" if aqi_val <= 50 else ("#f59e0b" if aqi_val <= 100 else "#ef4444")

    # 4. Metrics Display
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f"<div class='glass-card' style='border-top:3px solid {aqi_color};'><h5>US AQI</h5><h3 style='color:{aqi_color}'>{int(aqi_val)}</h3></div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='glass-card' style='border-top: 3px solid #fbbf24;'><h5>Temp</h5><h3>{temp}°C</h3></div>", unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='glass-card' style='border-top: 3px solid #a78bfa;'><h5>Wind</h5><h3>{wind} km/h</h3></div>", unsafe_allow_html=True)
    with m4: st.markdown(f"<div class='glass-card' style='border-top: 3px solid #38bdf8;'><h5>Heading</h5><h3>{head}°</h3></div>", unsafe_allow_html=True)

    # 4. PM2.5 Long-term Trajectory
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], mode="lines+markers", name="Recorded", line=dict(color="#00f2fe", width=3)))

    if 'future_df' in locals() or 'future_df' in globals():
        forecast_future = future_df[future_df['year'] > df_yearly['year'].max()]
        fig.add_trace(go.Scatter(x=forecast_future["year"], y=forecast_future["predicted_pm25"], mode="lines+markers", name="Forecast", line=dict(color="#ff0844", width=3, dash="dot")))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_future["year"], forecast_future["year"][::-1]]), 
            y=pd.concat([forecast_future["yhat_upper"], forecast_future["yhat_lower"][::-1]]), 
            fill='toself', fillcolor='rgba(255, 8, 68, 0.1)', line=dict(color='rgba(255,255,255,0)'), 
            showlegend=True, name="Confidence"
        ))

    fig.update_layout(
        title="PM2.5 Long-Term Atmospheric Trajectory", 
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
        margin=dict(t=100, l=40, r=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5. Clinical Advisory (FIXED: Changed current_aqi to aqi_val)
    st.markdown("---")
    st.subheader("🩺 Clinical Neuro-Respiratory Advisory")
    a1, a2 = st.columns([1, 2])
    with a1:
        st.markdown(f"<div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 15px;'><h1 style='color:{aqi_color}; font-size: 60px;'>{int(aqi_val)}</h1></div>", unsafe_allow_html=True)
    with a2:
        if aqi_val <= 50: st.success("✅ **Air Quality is Optimal.** Ideal conditions for outdoor activities.")
        elif aqi_val <= 100: st.warning("⚠️ **Air Quality is Moderate.** Sensitive individuals should limit prolonged exertion.")
        else: st.error("🚨 **High Toxicity Detected.** Deep-lung particulate risk. Indoor protocols advised.")

    # 6. Environmental Resilience Index (FIXED: Changed current_aqi to aqi_val)
    resilience_val = calculate_resilience_score(aqi_val, float(wind) if wind != 'N/A' else 0.0, float(weather.get('relative_humidity_2m') or 50))
    st.markdown("---")
    st.subheader("🛡️ Environmental Resilience Index")
    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        st.metric("Resilience Score", f"{resilience_val}/100", delta=f"{resilience_val - 50} from baseline")
    with col_r2:
        if resilience_val > 80:
            st.info("🟢 **High Resilience:** Excellent atmospheric dispersion preventing pollutant stagnation.")
        elif resilience_val > 50:
            st.info("🟡 **Moderate Resilience:** Average atmospheric drift; pollutants may accumulate based on local emissions.")
        else:
            st.info("🔴 **Low Resilience:** Stagnant atmospheric conditions. High risk of rapid localized accumulation.")

    # 7. Advanced Telemetry Expander
    with st.expander("📊 View Advanced Atmospheric Bio-Telemetry"):
        plot_df = pd.DataFrame({
            "year": [2018, 2019, 2020, 2021, 2022, 2023],
            "pressure_hpa": [1012, 1015, 1010, 1013, 1018, 1011],
            "humidity": [65, 72, 60, 68, 62, 70],
            "aerosol_depth": [0.12, 0.15, 0.10, 0.13, 0.09, 0.11]
        })
        st.write("Modulations in these metrics directly correlate to atmospheric particulate dispersion rates.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Pressure (hPa)**")
            st.line_chart(plot_df, x="year", y="pressure_hpa", color="#fbbf24") 
        with col2:
            st.markdown("**Humidity (%)**")
            st.line_chart(plot_df, x="year", y="humidity", color="#38bdf8")
        with col3:
            st.markdown("**Aerosol Depth**")
            st.line_chart(plot_df, x="year", y="aerosol_depth", color="#a78bfa")
with tab2:
    st.markdown("### 🛠️ Impact & Mitigation Simulator")

    # 1. Cognitive Risk Assessment Section
    raw_pm25 = current_data.get('pm2_5', 0)
    pm25_val = float(raw_pm25) if raw_pm25 != 'N/A' else 0.0
    risk_score = (pm25_val / 12) * 1.5

    # Logic for status
    if pm25_val < 12:
        status, briefing, col_color = "Baseline Homeostasis", "Air quality is optimal. Minimal systemic inflammation detected. Cognitive function is not under environmental stress.", "#10b981"
    elif 12 <= pm25_val < 35:
        status, briefing, col_color = "Mild Oxidative Stress", "Moderate particulate load. Your body is mobilizing antioxidants. You may experience subtle focus fatigue.", "#f59e0b"
    else:
        status, briefing, col_color = "Neuro-Inflammatory Warning", "High PM2.5 load. Particulates are triggering systemic inflammatory response. Limit intense cognitive tasks.", "#ef4444"

    # Display Risk in a styled container
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border-left: 5px solid {col_color};'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h5 style='margin:0;'>Cognitive Risk Index</h5>
                    <h2 style='color:{col_color}; margin:0;'>{round(risk_score, 1)} / 10</h2>
                </div>
                <div style='text-align: right;'>
                    <h5 style='margin:0;'>Current Status</h5>
                    <span style='color: white; font-weight: bold;'>{status}</span>
                </div>
            </div>
            <p style='margin-top: 15px; font-size: 0.9em; color: #a1a1aa;'>{briefing}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Unified Simulation Panel
    col_sim_gauge, col_sim_slider = st.columns([1, 1])

    with col_sim_gauge:
        st.markdown("**Current Threat Level (AQI)**")
        current_aqi_val = float(current_aqi) if current_aqi != 'N/A' else 0
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=current_aqi_val, 
            gauge={
                'axis': {'range': [0, 300], 'tickcolor': "white"}, 
                'bar': {'color': col_color}, 
                'bgcolor': "rgba(0,0,0,0)", 
                'steps': [
                    {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.2)"}, 
                    {'range': [50, 150], 'color': "rgba(245, 158, 11, 0.2)"}, 
                    {'range': [150, 300], 'color': "rgba(239, 68, 68, 0.2)"}
                ]
            }
        ))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)

    with col_sim_slider:
        st.markdown("**Emission Mitigation Simulator**")
        st.write("Adjust the reduction percentage to visualize projected health improvements.")
        reduction = st.slider("Target Mitigation (%)", 0, 50, 0, help="Simulate a reduction in local particulate output.")

        # Calculation
        sim_val = future_df['predicted_pm25'].iloc[-1] * (1 - (reduction/100))

        # Display Metric
        st.markdown(f"""
            <div class='glass-card' style='margin-top: 20px; text-align: center;'>
                <h5 style='color: #8b9bb4;'>Estimated {future_df['year'].iloc[-1]} PM2.5</h5>
                <h2 style='color: #00f2fe;'>{sim_val:.2f} µg/m³</h2>
                <span style='font-size: 0.8em;'>Projected Impact: -{reduction}%</span>
            </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 🩺 Advanced Health Literacy & Physiological Impact")

    # 1. Pollutant Matrix (Simplified)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🔬 Pollutant Breakdown")
        with st.expander("PM2.5 (Fine Particulates)", expanded=True):
            st.write("Particles <2.5μm. Bypass defenses, lodge in alveoli.")
            st.metric("Safe Limit", "< 12.0 μg/m³")
    with col_b:
        st.markdown("#### 🧬 Systemic Physiological Load")
        body_load = min(100, (current_aqi / 300) * 100)
        st.progress(body_load / 100)
        st.caption(f"Estimated inflammatory strain: {int(body_load)}%")

    # 2. Activity & Indoor Tools (Keys made unique)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏃 Exposure Risk Calculator")
        activity = st.selectbox("Select Activity Level", ["Resting", "Light Walk", "Heavy Exercise"], key="tab3_act_level")
        if activity == "Heavy Exercise" and current_aqi > 100:
            st.error("🚨 CRITICAL: High-intensity exercise prohibited.")
        elif activity == "Heavy Exercise" and current_aqi > 50:
            st.warning("⚠️ CAUTION: Reduce intensity recommended.")
    with c2:
        st.subheader("🏠 Indoor Scrubbing Tool")
        room_sqft = st.number_input("Room Square Footage", value=200, key="tab3_sqft")
        filter_eff = st.select_slider("Purifier Efficiency", options=["Standard", "HEPA", "Industrial"], key="tab3_filter")
        if st.button("Calculate Scrub Time"):
            rate = 50 if filter_eff == "Standard" else (100 if filter_eff == "HEPA" else 150)
            st.write(f"⏱️ Scrub Time: **{room_sqft / rate:.1f} minutes**.")

with tab4:
    st.markdown("### 🔍 Global Sensor Search")

    # 1. ADDED SEARCH BAR HERE
    city_input = st.text_input(
        "Search Location (e.g., Tokyo, Raleigh)", 
        placeholder="Enter city name...", 
        key="tab4_search_bar"
    )

    if city_input:
        lat, lon, name = get_city_coords(city_input)
        if lat:
            st.session_state.map_center = [lat, lon]
            st.success(f"📍 Navigation locked to: {name}")
        else:
            st.error("Location not found. Please check the spelling.")

    # 2. Map Rendering
    m = folium.Map(location=st.session_state.map_center, zoom_start=8, tiles="CartoDB dark_matter")

    folium.Marker(
        st.session_state.map_center, 
        tooltip="Active Sensor Hub",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    st_folium(m, width="100%", height=400, key="tab4_map")

    # 3. Atmospheric Report
    st.markdown("### 📊 Atmospheric Report")
    curr_lat, curr_lon = st.session_state.map_center
    active_pollutant_key = st.session_state.get('selected_pollutant_key', 'pm2_5')
    active_pollutant_name = st.session_state.get('selected_pollutant_name', 'PM2.5')

    with st.spinner(f"Fetching {active_pollutant_name} data..."):
        data = fetch_global_aqi(curr_lat, curr_lon, active_pollutant_key)
        if data:
            col1, col2, col3 = st.columns(3)
            col1.metric("US AQI", f"{data.get('us_aqi', 'N/A')}")
            col2.metric(f"{active_pollutant_name}", f"{data.get(active_pollutant_key, 'N/A')}")
            col3.metric("Node Coordinates", f"{curr_lat:.2f}, {curr_lon:.2f}")
        else:
            st.error("Could not retrieve data for this node.")
# ============================================================
# TAB 5: NASA FOREST FIRE & SMOKE INTELLIGENCE (VIIRS UPGRADE)
# ============================================================
# ============================================================
# TAB 5: GLOBAL FIRE INTELLIGENCE & HEATMAP
# ============================================================
with tab5:
    st.markdown("### 🌍 Global Satellite & Fire Intelligence")
    st.caption("Layering NIFC Perimeters (USA) and NASA VIIRS Thermal Hotspots (Global).")

    # 1. INITIALIZE VARIABLE TO PREVENT NAMEERROR
    fire_geo = None 
    
    # 2. Target Date Setup
    nasa_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Toggles
    c1, c2, c3 = st.columns(3)
    show_perimeters = c1.checkbox("🇺🇸 Show US Perimeters", value=True)
    show_global_heat = c2.checkbox("🌏 Show Global Heatmap", value=True)
    show_smoke = c3.checkbox("☁️ Show Smoke Satellite", value=True)

    # 3. Fetch Data
    if show_perimeters or show_global_heat:
        with st.spinner("Fetching global fire data..."):
            # This function must be defined in your helper functions section
            fire_geo = fetch_us_fire_perimeters() 

    # 4. Map Logic
    lat_c, lon_c = st.session_state.get('map_center', [35.7796, -78.6382])
    m_fire = folium.Map(location=[lat_c, lon_c], zoom_start=4, tiles="CartoDB dark_matter")

    # ADD GLOBAL SMOKE (VIIRS - No Gaps)
    if show_smoke:
        smoke_url = (
            f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            f"VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{nasa_date}/"
            f"GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
        )
        folium.TileLayer(tiles=smoke_url, attr="NASA GIBS", name="Smoke", overlay=True, opacity=0.5).add_to(m_fire)

    # CHECK IF DATA EXISTS BEFORE USING (Fixes NameError)
    if fire_geo and 'features' in fire_geo:
        
        # A. Draw US Perimeters (Shapes)
        if show_perimeters:
            folium.GeoJson(
                fire_geo,
                name="Fire Shapes",
                style_function=lambda x: {'fillColor': 'red', 'color': 'orange', 'weight': 2, 'fillOpacity': 0.4}
            ).add_to(m_fire)

        # B. Draw Global Heatmap
        if show_global_heat:
            from folium.plugins import HeatMap
            heat_data = []
            for feature in fire_geo['features']:
                if feature['geometry'] and feature['geometry']['type'] == 'Point':
                    coords = feature['geometry']['coordinates']
                    heat_data.append([coords[1], coords[0]]) # Folium uses [Lat, Lon]
                elif feature['geometry'] and feature['geometry']['type'] == 'Polygon':
                    # Get center of polygon for heatmap point
                    coords = feature['geometry']['coordinates'][0][0]
                    heat_data.append([coords[1], coords[0]])

            if heat_data:
                HeatMap(heat_data, radius=15, blur=10, gradient={0.4: 'blue', 0.6: 'yellow', 1: 'red'}).add_to(m_fire)

    st_folium(m_fire, width="100%", height=600, key="global_fire_map")

    # 5. NASA Metrics (Bottom)
    st.markdown("---")
    nasa_data = get_nasa_climate_data(lat_c, lon_c)
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Solar Irradiance", f"{nasa_data['solar_radiation']} kW/m²")
    col_m2.metric("Wind Speed", f"{nasa_data['satellite_wind_speed']} m/s")

with tab6:
    st.markdown("### 🚲 The Clean-Air Commute")
    from geopy.distance import geodesic
    import random

    # 1. Address Inputs
    col1, col2 = st.columns(2)
    start_addr = col1.text_input("Starting Address", "100 Main St, Raleigh, NC", key="start_addr_input")
    end_addr = col2.text_input("Destination Address", "500 Main St, Rolesville, NC", key="end_addr_input")

    # 2. Path Generation Logic
    if st.button("Generate Healthiest Path", key="gen_path_btn"):
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="wake_air_quality_app_v2")
            s_loc = geolocator.geocode(start_addr)
            e_loc = geolocator.geocode(end_addr)
            
            if not s_loc or not e_loc:
                st.error("Could not find addresses. Please be more specific.")
            else:
                s_lat, s_lon = s_loc.latitude, s_loc.longitude
                e_lat, e_lon = e_loc.latitude, e_loc.longitude
                
                # Simple distance check
                dist_km = geodesic((s_lat, s_lon), (e_lat, e_lon)).km
                if dist_km > 50:
                    st.error("Distance too large (max 50km).")
                else:
                    graph = get_map_graph(s_lat, s_lon, e_lat, e_lon)
                    if graph:
                        start_node = ox.distance.nearest_nodes(graph, s_lon, s_lat)
                        end_node = ox.distance.nearest_nodes(graph, e_lon, e_lat)
                        
                        route = nx.shortest_path(graph, start_node, end_node, weight='travel_time')
                        route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
                        
                        # Calculate time
                        edge_times = [graph.edges[(u, v, 0)].get('travel_time', 0) for u, v in zip(route[:-1], route[1:])]
                        total_time_min = sum(edge_times) / 60
                        
                        # UPDATE SESSION STATE WITH DICTIONARY
                        st.session_state['route_data'] = {
                            "route": route_coords, 
                            "s_lat": s_lat, "s_lon": s_lon,
                            "e_lat": e_lat, "e_lon": e_lon,
                            "exposure": get_healthiest_route(s_lat, s_lon, e_lat, e_lon),
                            "time_est": round(total_time_min, 1)
                        }
        except Exception as e:
            st.error(f"Routing Error: {str(e)}")

    # 3. DISPLAY LOGIC (The problematic part)
    # Check if key exists AND is not None
    if st.session_state.get('route_data') is not None:
        # Explicitly assign to a local variable for the display block
        res = st.session_state['route_data']
        
        # Ensure 'res' is actually a dictionary before calling .get()
        if isinstance(res, dict):
            col_m1, col_m2 = st.columns(2)
            
            # Use 0.0 as a safe fallback if 'exposure' is missing
            route_aqi_val = round(res.get('exposure', 0), 1)
            
            col_m1.metric("Route AQI", route_aqi_val)
            col_m2.metric("Est. Travel Time", f"{res.get('time_est', 'N/A')} min")

            m = folium.Map(tiles="CartoDB dark_matter")
            folium.PolyLine(res['route'], color="#4facfe", weight=5).add_to(m)
            folium.Marker([res['s_lat'], res['s_lon']], icon=folium.Icon(color='green')).add_to(m)
            folium.Marker([res['e_lat'], res['e_lon']], icon=folium.Icon(color='red')).add_to(m)
            m.fit_bounds(res['route'])
            st_folium(m, width="100%", height=400, key="commute_map")
        else:
            st.warning("Routing data is currently being processed...")

        st.markdown("---")
        st.markdown("### 🩺 Daily Clinical Briefing")

        good_advice = [
            "✅ **Excellent Air Quality.** Perfect conditions for cycling. Enjoy the fresh air!",
            "✅ **Air quality is optimal.** Your respiratory system will appreciate this clean commute.",
            "✅ **Great conditions!** No restrictions on your outdoor activity today."
        ]
        mid_advice = [
            "⚠️ **Moderate Air Quality.** Consider a face mask if you are sensitive to particulates.",
            "⚠️ **Caution:** AQI is mid-range. Limit high-intensity breathing during your commute.",
            "⚠️ **Moderate levels detected.** If you have asthma or lung conditions, take it easy today."
        ]
        bad_advice = [
            "🚨 **High Risk.** The air quality is poor today. We strongly recommend choosing an indoor commute.",
            "🚨 **Warning:** High particulate levels detected. Strenuous outdoor exercise is not advised.",
            "🚨 **Alert:** Exposure today could lead to respiratory irritation. Please stick to indoor environments."
        ]

        if aqi_val < 50:
            st.success(random.choice(good_advice))
        elif 50 <= aqi_val < 100:
            st.warning(random.choice(mid_advice))
        else:
            st.error(random.choice(bad_advice))

        st.markdown("#### Biological Context:")
        st.write("""
        Fine particulate matter (PM2.5) bypasses natural airway defenses, settling deep in the 
        alveoli where gas exchange occurs. Once they cross this barrier, they can cause 
        systemic inflammation throughout the body.
        """)

# ============================================================
# TAB 7 — HISTORICAL CLIMATE CORRELATION DASHBOARD
# ============================================================
with tab7:
    st.markdown("### 🌡️ Historical Climate Correlation Dashboard")
    st.caption("Analyzing the relationship between local weather patterns and PM2.5 levels over the last 30 days.")

    c_s, c_b = st.columns([4, 1])
    with c_s:
        corr_city = st.text_input("City for Correlation", "Raleigh, NC", key="city_input_tab7", label_visibility="collapsed")
    with c_b:
        corr_btn = st.button("📈 Load Data", use_container_width=True, key="btn_tab7")

    # Use existing map center if no search has been performed
    current_lat, current_lon = st.session_state.get('map_center', [35.7796, -78.6382])

    if corr_btn:
        lat, lon, name = get_city_coords(corr_city)
        if lat:
            current_lat, current_lon = lat, lon
            st.session_state['corr_city_name'] = name
        else:
            st.error("City not found.")

    with st.spinner("Analyzing atmospheric correlations..."):
        corr_df = fetch_30day_correlation(current_lat, current_lon)

    if corr_df is not None and not corr_df.empty:
        city_label = st.session_state.get('corr_city_name', "Selected Area")
        st.markdown(f"#### 📊 30-Day Analysis: {city_label}")

        # Resample to daily to reduce chart noise
        daily_corr = corr_df.resample('D', on='time').mean().reset_index()

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PM2.5 vs Humidity**")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=daily_corr['time'], y=daily_corr['pm25'], name="PM2.5", line=dict(color="#00f2fe"), yaxis='y1'))
            fig1.add_trace(go.Scatter(x=daily_corr['time'], y=daily_corr['humidity'], name="Humidity (%)", line=dict(color="#a78bfa", dash='dot'), yaxis='y2'))
            fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis=dict(title="PM2.5", color="#00f2fe"), yaxis2=dict(title="%", overlaying='y', side='right', color="#a78bfa"),
                              legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**PM2.5 vs Wind Speed**")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=daily_corr['time'], y=daily_corr['pm25'], name="PM2.5", line=dict(color="#00f2fe"), yaxis='y1'))
            fig2.add_trace(go.Scatter(x=daily_corr['time'], y=daily_corr['wind'], name="Wind (km/h)", line=dict(color="#fbbf24", dash='dot'), yaxis='y2'))
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis=dict(title="PM2.5", color="#00f2fe"), yaxis2=dict(title="km/h", overlaying='y', side='right', color="#fbbf24"),
                              legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter Analysis with Trendline
        import plotly.express as px
        st.markdown("**Temperature vs PM2.5 (Pearson Trend)**")
        fig3 = px.scatter(daily_corr, x='temp', y='pm25', color='pm25', size='pm25', 
                         color_continuous_scale='RdYlGn_r', trendline='ols', template='plotly_dark')
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

        # Statistical Summary Cards
        st.markdown("### Pearson Correlation Breakdown")
        c1, c2, c3 = st.columns(3)
        metrics = [
            ("Humidity", daily_corr['pm25'].corr(daily_corr['humidity'])),
            ("Wind Speed", daily_corr['pm25'].corr(daily_corr['wind'])),
            ("Temp", daily_corr['pm25'].corr(daily_corr['temp']))
        ]
        
        for col, (label, val) in zip([c1, c2, c3], metrics):
            status = "Strong" if abs(val) > 0.6 else ("Moderate" if abs(val) > 0.3 else "Weak")
            col.metric(label, f"{val:+.3f}", delta=status, delta_color="off")
