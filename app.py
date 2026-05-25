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
import gspread
from google.oauth2.service_account import Credentials
from googletrans import Translator



# ============================================
# INITIALIZATION & STATE
# ============================================


def create_base_map(lat, lon, zoom=12):
    return folium.Map(location=[lat, lon], zoom_start=zoom, tiles="CartoDB dark_matter")

def add_heatmap_to_map(m, hotspots):
    from folium.plugins import HeatMap
    HeatMap(hotspots, radius=20, blur=15).add_to(m)
    return m

def get_db_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    
    # 1. Open the Spreadsheet by Name
    spreadsheet = client.open("AirQualityHazards")
    
    # 2. Return the SPECIFIC worksheet (change "Sheet1" if yours is named differently)
    return spreadsheet.worksheet("Sheet1") 

def get_global_hazards():
    try:
        # Get the worksheet object
        worksheet = get_db_client()
        # Get the data
        data = worksheet.get_all_records()
        return data
    except Exception as e:
        st.error(f"Error fetching hazards: {e}")
        return []

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
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

@st.cache_data(ttl=3600)
def fetch_7day_forecast(lat, lon):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&daily=us_aqi_max&timezone=auto&forecast_days=7"
    try:
        response = requests.get(url, timeout=5) # Added timeout
        if response.status_code == 200:
            data = response.json()
            daily = data.get("daily", {})
            if "time" in daily and "us_aqi_max" in daily:
                return pd.DataFrame({
                    "date": pd.to_datetime(daily["time"]),
                    "aqi": daily["us_aqi_max"]
                })
        return pd.DataFrame() # Returns empty if API fails
    except Exception:
        return pd.DataFrame() # Returns empty if connection fails

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

@st.cache_data(ttl=3600)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    try:
        return requests.get(url, timeout=10).json().get("current", None)
    except Exception: 
        return None
# --- END CLEANED UP SECTION ---

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

# --- FIX: Ensure live_weather is always a dictionary ---
with st.spinner("Initializing Atmospheric Sensors..."):
    # This ensures live_weather is never None, but a safe empty dict instead
    live_weather_raw = fetch_live_weather(35.7796, -78.6382)
    live_weather = live_weather_raw if live_weather_raw is not None else {}

@st.cache_data(ttl=3600)
def fetch_7day_forecast(lat, lon):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&daily=us_aqi_max&timezone=auto&forecast_days=7"
    try:
        response = requests.get(url).json()
        daily = response.get("daily", {})
        return pd.DataFrame({
            "date": pd.to_datetime(daily.get("time")),
            "aqi": daily.get("us_aqi_max")
        })
    except:
        return pd.DataFrame()

# ============================================
# MAIN UI LAYOUT
# ============================================
st.markdown('<p class="title-gradient">Project AIR</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Project AIR (Atmospheric Intelligence & Response) is a Global Atmospheric PM2.5 Analytics Engine.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Telemetry & Forecasting", "🧠 Hypothetical Prediction Measure", "🩺 Health Literacy", "🛰️ Global Vector Map", "🌌NASA Forest Fire Intelligence", "🚲Clean-Air Commute"])

# --- TAB 1: OVERVIEW ---

# --- TAB 1: OVERVIEW ---
with tab1:
    # 1. High-Tech Console Search Input
    st.markdown("### 🌍 Regional Atmospheric & Bio-Telemetry Analysis")

    st.markdown("""
        <style>
        .search-box {
            background: linear-gradient(90deg, rgba(16,25,43,0.8) 0%, rgba(11,15,25,0.8) 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #00f2fe;
            margin-bottom: 25px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            city_input = st.text_input("Search Location", "Raleigh", key="tab1_city", label_visibility="collapsed", placeholder="Enter City or Coordinates...")
        with col_btn:
            if st.button("📡 Scan Region", use_container_width=True):
                lat, lon, name = get_city_coords(city_input)
                if lat:
                    st.session_state.map_center = [lat, lon]
                    st.session_state.current_data = fetch_global_aqi(lat, lon)
                    st.session_state.current_weather = fetch_live_weather(lat, lon)
                    st.rerun()
                else:
                    st.error("❌ Target lost.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. State Retrieval
    curr_lat, curr_lon = st.session_state.get('map_center', [35.7796, -78.6382])
    data = st.session_state.get('current_data', fetch_global_aqi(curr_lat, curr_lon))
    weather = st.session_state.get('current_weather', fetch_live_weather(curr_lat, curr_lon)) or {}
    current_aqi = float(data.get('us_aqi', 0))

    # Dynamic UI colors
    aqi_color = "#10b981" if current_aqi <= 50 else ("#f59e0b" if current_aqi <= 100 else ("#f97316" if current_aqi <= 150 else "#ef4444"))

    # 3. Metrics Display (Glowing Glass-card UI)
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f"<div class='glass-card' style='border-top: 3px solid {aqi_color};'><h5>US AQI</h5><h3 style='color:{aqi_color}'>{current_aqi}</h3></div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='glass-card' style='border-top: 3px solid #fbbf24;'><h5>Temp</h5><h3>{weather.get('temperature_2m', 'N/A')}°C</h3></div>", unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='glass-card' style='border-top: 3px solid #a78bfa;'><h5>Wind</h5><h3>{weather.get('wind_speed_10m', 'N/A')} km/h</h3></div>", unsafe_allow_html=True)
    with m4: st.markdown(f"<div class='glass-card' style='border-top: 3px solid #38bdf8;'><h5>Heading</h5><h3>{weather.get('wind_direction_10m', 'N/A')}°</h3></div>", unsafe_allow_html=True)

    # --- ALERT THRESHOLD LOGIC (NOW PROPERLY INDENTED) ---
    if current_aqi > alert_threshold:
        st.error(f"⚠️ ALERT: Current AQI ({current_aqi}) exceeds your defined threshold of {alert_threshold}!")
        st.warning("Recommendation: Engage indoor air purification protocols immediately.")
    else:
        st.success(f"✅ AQI ({current_aqi}) is within your specified safety threshold ({alert_threshold}).")

    # 5. Clinical Advisory
    st.markdown("---")
    st.subheader("🩺 Clinical Neuro-Respiratory Advisory")
    a1, a2 = st.columns([1, 2])
    with a1:
        st.markdown(f"<div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 15px;'><h1 style='color:{aqi_color}; font-size: 60px;'>{current_aqi}</h1></div>", unsafe_allow_html=True)
    with a2:
        if current_aqi <= 50: st.success("✅ **Air Quality is Optimal.** Ideal conditions for outdoor activities.")
        elif current_aqi <= 100: st.warning("⚠️ **Air Quality is Moderate.** Sensitive individuals should limit prolonged exertion.")
        else: st.error("🚨 **High Toxicity Detected.** Deep-lung particulate risk. Indoor protocols advised.")

    # 6. Environmental Resilience Index
    resilience_val = calculate_resilience_score(current_aqi, weather.get('wind_speed_10m', 0), weather.get('relative_humidity_2m', 50))
    st.markdown("---")
    st.subheader("🛡️ Environmental Resilience Index")
    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        st.metric("Resilience Score", f"{resilience_val}/100", delta=f"{resilience_val - 50} from baseline", delta_color="normal" if resilience_val >= 50 else "inverse")
    with col_r2:
        if resilience_val > 80:
            st.info("🟢 **High Resilience:** Excellent atmospheric dispersion preventing pollutant stagnation.")
        elif resilience_val > 50:
            st.info("🟡 **Moderate Resilience:** Average atmospheric drift; pollutants may accumulate based on local emissions.")
        else:
            st.info("🔴 **Low Resilience:** Stagnant atmospheric conditions. High risk of rapid localized accumulation.")

    # 7. Advanced Telemetry Expander
    st.markdown("---")
    plot_df = pd.DataFrame({
        "year": [2018, 2019, 2020, 2021, 2022, 2023],
        "pressure_hpa": [1012, 1015, 1010, 1013, 1018, 1011],
        "humidity": [65, 72, 60, 68, 62, 70],
        "aerosol_depth": [0.12, 0.15, 0.10, 0.13, 0.09, 0.11]
    })

    with st.expander("📊 View Advanced Atmospheric Bio-Telemetry"):
        st.write("Modulations in these metrics directly correlate to atmospheric particulate dispersion rates.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Atmospheric Pressure (hPa)**")
            st.line_chart(plot_df, x="year", y="pressure_hpa", color="#fbbf24") 
        with col2:
            st.markdown("**Relative Humidity (%)**")
            st.line_chart(plot_df, x="year", y="humidity", color="#38bdf8")
        with col3:
            st.markdown("**Aerosol Optical Depth**")
            st.line_chart(plot_df, x="year", y="aerosol_depth", color="#a78bfa")

# --- TAB 2: ANALYTICS ---
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

        # Updated logic: Use current_aqi as the baseline for the simulation 
        # since we no longer have the long-term forecast dataframe
        sim_val = current_aqi * (1 - (reduction/100))

        # Display Metric
        st.markdown(f"""
            <div class='glass-card' style='margin-top: 20px; text-align: center;'>
                <h5 style='color: #8b9bb4;'>Projected PM2.5 Level</h5>
                <h2 style='color: #00f2fe;'>{sim_val:.2f} µg/m³</h2>
                <span style='font-size: 0.8em;'>Projected Impact: -{reduction}%</span>
            </div>
        """, unsafe_allow_html=True)

st.subheader("🧬 Biological Impact: Your Lung Age")

if 'map_center' in st.session_state:
    # Get the real-time AQI from your sensor data
    # We use a base_aqi of 50 as the "Clean Air Standard"
    local_aqi = current_aqi if 'current_aqi' in globals() else 50
    
    col_a, col_b = st.columns(2)
    user_age = col_a.number_input("Your Actual Age", 18, 100, 30, key="age_in")
    years_resident = col_b.number_input("Years living in this city", 0, 80, 5, key="years_in")
    
    # NEW LOGIC: Cumulative Exposure Burden (CEB)
    # 1. Base Impact: (AQI/100) * 0.5 years per year of residency
    # 2. Threshold Penalty: If AQI > 100 (Unhealthy), multiply impact by 1.5x
    threshold_multiplier = 1.5 if local_aqi > 100 else 1.0
    exposure_impact = (local_aqi / 100) * 0.5 * threshold_multiplier
    
    lung_age = user_age + (years_resident * exposure_impact)
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Calculated Biological Lung Age", f"{lung_age:.1f} years")
    
    # More realistic 'danger' logic
    age_gap = lung_age - user_age
    if age_gap > 10:
        res_col2.error(f"🚨 ALERT: Chronic exposure has aged your lungs by {age_gap:.1f} years.")
    elif age_gap > 5:
        res_col2.warning(f"⚠️ CAUTION: Your lung health is {age_gap:.1f} years older than your actual age.")
    else:
        res_col2.success("✅ Exposure index within acceptable clinical range.")
        
    st.caption("Calculated based on chronic exposure modeling. High AQI significantly accelerates biological lung aging.")

# 4. 7-Day Atmospheric Outlook
    st.markdown("### 📅 7-Day Atmospheric Outlook")
    # Call your new API function (make sure it's defined at the top of your file)
    forecast_df = fetch_7day_forecast(curr_lat, curr_lon)

    if not forecast_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["aqi"], 
            mode="lines+markers", 
            name="AQI Forecast",
            line=dict(color="#00f2fe", width=4),
            fill='tozeroy', 
            fillcolor='rgba(0, 242, 254, 0.1)'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Forecast data currently unavailable.")

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

    # 1. Search Bar
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
    
    # Active Sensor Marker
    folium.Marker(
        st.session_state.map_center, 
        tooltip="Active Sensor Hub",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    # 3. Community Hazard Reporting
    st.subheader("🚩 Community Hazards")
    if 'hazard_reports' not in st.session_state:
        st.session_state.hazard_reports = []

    if st.button("🚩 Report Poor Air at Current Location", key="report_btn"):
        st.session_state.hazard_reports.append(st.session_state.map_center)
        st.toast("Report submitted! Your community thanks you.", icon="✅")

    # Draw Hazard Pins
    for report in st.session_state.hazard_reports:
        folium.Marker(
            report,
            popup="Community Reported Hazard",
            icon=folium.Icon(color='red', icon='warning-sign')
        ).add_to(m)
    
    st_folium(m, width="100%", height=400, key="tab4_map")

    # 4. Atmospheric Report
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
# --- TAB 5: SPACE INTEL ---
with tab5:
    st.markdown("### 🌍 Global Satellite Intelligence For Forest Fires")

    # Dynamically set the date to today (2026-05-24)
    today_str = datetime.now().strftime("%Y-%m-%d")
    tile_url = f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_Thermal_Anomalies_375m/default/{today_str}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.png"

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.5),
        layers=[
            pdk.Layer(
                "TileLayer",
                tile_url,
                opacity=0.8
            ),
        ],
    ))
    st.caption(f"Map: NASA VIIRS Thermal Anomalies for {today_str}.")

    st.markdown("### 🛰️ Climate Metrics")
    # Using your existing coordinates as a default
    nasa_data = get_nasa_climate_data(35.7796, -78.6382)
    col1, col2 = st.columns(2)
    col1.metric("Surface Solar Irradiance", f"{nasa_data['solar_radiation']} kW/m²")
    col2.metric("Satellite Wind Velocity", f"{nasa_data['satellite_wind_speed']} m/s")
    
with tab6:
    st.markdown("### 🚲 The Clean-Air Commute")
    from geopy.distance import geodesic
    import random

    col1, col2 = st.columns(2)
    start_addr = col1.text_input("Starting Address", "100 Main St, Raleigh, NC", key="start_addr")
    end_addr = col2.text_input("Destination Address", "500 Main St, Rolesville, NC", key="end_addr")

    if st.button("Generate Healthiest Path"):
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="wake_air_quality_app")
            s_loc = geolocator.geocode(start_addr)
            e_loc = geolocator.geocode(end_addr)
            if not s_loc or not e_loc:
                st.error("Could not find addresses. Please be more specific.")
            else:
                s_lat, s_lon = s_loc.latitude, s_loc.longitude
                e_lat, e_lon = e_loc.latitude, e_loc.longitude
                dist_km = geodesic((s_lat, s_lon), (e_lat, e_lon)).km
                if dist_km > 50:
                    st.error("Distance too large (max 50km).")
                else:
                    graph = get_map_graph(s_lat, s_lon, e_lat, e_lon)
                    if graph:
                        start_node = ox.distance.nearest_nodes(graph, s_lon, s_lat)
                        end_node = ox.distance.nearest_nodes(graph, e_lon, e_lat)
                        try:
                            route = nx.shortest_path(graph, start_node, end_node, weight='travel_time')
                            route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
                            total_time_min = sum(graph.edges[(u, v, 0)].get('travel_time', 0) for u, v in zip(route[:-1], route[1:])) / 60
                            st.session_state.route_data = {
                                "route": route_coords, "s_lat": s_lat, "s_lon": s_lon,
                                "e_lat": e_lat, "e_lon": e_lon,
                                "exposure": get_healthiest_route(s_lat, s_lon, e_lat, e_lon),
                                "time_est": round(total_time_min, 1)
                            }
                        except Exception:
                            st.error("No path found.")
        except Exception as e:
            st.error(f"Address error: {e}")

    if 'route_data' in st.session_state:
        data = st.session_state.route_data
        col1, col2 = st.columns(2)
        aqi_val = round(data.get('exposure', 0), 1)
        col1.metric("Route AQI", aqi_val)
        col2.metric("Est. Travel Time", f"{data.get('time_est', 'N/A')} min")

        m = folium.Map(tiles="CartoDB dark_matter")
        folium.PolyLine(data['route'], color="#4facfe", weight=5).add_to(m)
        folium.Marker([data['s_lat'], data['s_lon']], icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([data['e_lat'], data['e_lon']], icon=folium.Icon(color='red')).add_to(m)
        m.fit_bounds(data['route'])
        st_folium(m, width="100%", height=400)

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
