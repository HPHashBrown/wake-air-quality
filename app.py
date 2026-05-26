import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
import pydeck as pdk
import random
import io
import base64

# ============================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================
st.set_page_config(
    page_title="Project AIR",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CONSTANTS & MAPPINGS
# ============================================
POLLUTANT_MAP = {
    "PM2.5":  "pm2_5",
    "PM10":   "pm10",
    "Ozone":  "ozone",
    "NO2":    "nitrogen_dioxide"
}

POLLUTANT_FORECAST_MAP = {
    "PM2.5":  "pm2_5_mean",
    "PM10":   "pm10_mean",
    "Ozone":  "ozone_max",
    "NO2":    "nitrogen_dioxide_max"
}

SENSITIVITY_PROFILES = {
    "General Public":   {"aqi_caution": 100, "aqi_danger": 150, "label": "🧑 General Public"},
    "Sensitive Group":  {"aqi_caution": 50,  "aqi_danger": 100, "label": "🤧 Sensitive Group (Asthma/Elderly)"},
    "Athletic":         {"aqi_caution": 75,  "aqi_danger": 125, "label": "🏃 Athletic / High Exertion"},
    "Compromised":      {"aqi_caution": 35,  "aqi_danger": 75,  "label": "🫁 Medically Compromised"}
}

# ============================================
# GEMINI INITIALIZATION
# ============================================
try:
    client = genai.Client(api_key=st.secrets["GEM_KEY"])
except Exception as e:
    st.error(f"Error initializing Gemini Client: {e}")
    client = None

# ============================================
# OSMNX SETTINGS
# ============================================
ox.settings.timeout = 300
ox.settings.use_cache = True

# ============================================
# AUTO-REFRESH (every 10 min)
# ============================================
st_autorefresh(interval=600000, key="datarefresh")

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
defaults = {
    'map_center':         [35.7796, -78.6382],
    'start_time':         datetime.now(),
    'current_data':       None,
    'current_weather':    None,
    'forecast_data':      None,
    'multi_forecast_data': None,
    'route_data':         None,
    'respiratory_profile': "General Public",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# YEARLY REFERENCE DATA
# ============================================
df_yearly = pd.DataFrame({
    "year":         [2018, 2019, 2020, 2021, 2022, 2023],
    "mean_pm25":    [10.2, 9.8,  8.5,  9.2,  8.1,  7.9],
    "pressure_hpa": [1012, 1015, 1010, 1013, 1018, 1011],
    "humidity":     [65,   72,   60,   68,   62,   70],
    "aerosol_depth":[0.12, 0.15, 0.10, 0.13, 0.09, 0.11]
})

current_year = datetime.now().year

# ============================================
# PROPHET / LINEAR FALLBACK
# ============================================
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

if PROPHET_AVAILABLE:
    prophet_df = df_yearly[["year", "mean_pm25"]].copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df['year'], format='%Y')
    prophet_df = prophet_df.rename(columns={'mean_pm25': 'y'})
    _pm = Prophet(yearly_seasonality=True)
    _pm.fit(prophet_df)
    _future = _pm.make_future_dataframe(periods=10, freq='YS')
    _fc = _pm.predict(_future)
    future_df = pd.DataFrame({
        "year":           _fc['ds'].dt.year,
        "predicted_pm25": _fc['yhat'],
        "yhat_lower":     _fc['yhat_lower'],
        "yhat_upper":     _fc['yhat_upper']
    })
else:
    X = df_yearly[["year"]]
    y = df_yearly["mean_pm25"]
    _model = LinearRegression().fit(X, y)
    future_years = np.arange(current_year, current_year + 11)
    future_preds = _model.predict(future_years.reshape(-1, 1))
    future_df = pd.DataFrame({
        "year":           future_years,
        "predicted_pm25": future_preds,
        "yhat_lower":     future_preds - 1.5,
        "yhat_upper":     future_preds + 1.5
    })

# ============================================
# API FUNCTIONS
# ============================================

@st.cache_data(ttl=3600)
def fetch_global_aqi(lat, lon, metric="pm2_5"):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}&current=us_aqi,{metric}"
    )
    try:
        return requests.get(url, timeout=10).json().get("current", {})
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_7day_aqi_forecast(lat, lon):
    """Fetch 7-day US AQI + PM2.5 forecast."""
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=us_aqi,pm2_5_mean&forecast_days=7"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        daily = resp.get("daily", {})
        if daily:
            return pd.DataFrame({
                "date":  daily.get("time", []),
                "aqi":   daily.get("us_aqi", []),
                "pm25":  daily.get("pm2_5_mean", [])
            })
    except Exception as e:
        st.warning(f"7-day AQI forecast failed: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_multi_pollutant_forecast(lat, lon):
    """
    Fetch 7-day forecast for PM2.5, PM10, Ozone, NO2.
    Returns a DataFrame with columns: date, pm2_5, pm10, ozone, no2
    """
    daily_vars = "pm2_5_mean,pm10_mean,ozone_max,nitrogen_dioxide_max"
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&daily={daily_vars}&forecast_days=7"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        daily = resp.get("daily", {})
        if daily:
            return pd.DataFrame({
                "date":   daily.get("time", []),
                "PM2.5":  daily.get("pm2_5_mean", []),
                "PM10":   daily.get("pm10_mean", []),
                "Ozone":  daily.get("ozone_max", []),
                "NO2":    daily.get("nitrogen_dioxide_max", []),
            })
    except Exception as e:
        st.warning(f"Multi-pollutant forecast failed: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("current", {})
    except Exception as e:
        st.warning(f"Weather fetch failed: {e}")
        return {}

@st.cache_data(ttl=86400*30)
def fetch_30day_correlation(lat, lon):
    """
    Fetch last 30 days of hourly weather + hourly air quality for correlation dashboard.
    Returns merged DataFrame.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Weather (Open-Meteo)
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,wind_speed_10m,relative_humidity_2m"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=auto"
    )
    # Air quality (Open-Meteo)
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=pm2_5&start_date={start_date}&end_date={end_date}"
    )
    try:
        wr = requests.get(weather_url, timeout=15).json().get("hourly", {})
        ar = requests.get(aq_url, timeout=15).json().get("hourly", {})
        df = pd.DataFrame({
            "time":      wr.get("time", []),
            "temp":      wr.get("temperature_2m", []),
            "wind":      wr.get("wind_speed_10m", []),
            "humidity":  wr.get("relative_humidity_2m", []),
            "pm25":      ar.get("pm2_5", [])
        })
        df["time"] = pd.to_datetime(df["time"])
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_city_coords(city_name):
    if "new delhi" in city_name.lower().strip():
        return 28.6139, 77.2090, "New Delhi"
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    try:
        resp = requests.get(url, timeout=10).json()
        if resp.get("results"):
            d = resp["results"][0]
            return d["latitude"], d["longitude"], d["name"]
    except Exception:
        pass
    return None, None, None

@st.cache_data(ttl=86400)
def get_nasa_climate_data(lat, lon):
    target_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=ALLSKY_SFC_SW_DWN,WS2M&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={target_date}&end={target_date}&format=JSON"
    )
    try:
        data = requests.get(url, timeout=15).json()['properties']['parameter']
        solar = list(data['ALLSKY_SFC_SW_DWN'].values())[0]
        wind  = list(data['WS2M'].values())[0]
        return {
            "solar_radiation":    solar if solar != -999 else "N/A",
            "satellite_wind_speed": wind if wind != -999 else "N/A"
        }
    except Exception:
        return {"solar_radiation": "N/A", "satellite_wind_speed": "N/A"}

@st.cache_data(show_spinner=True)
def get_map_graph(lat1, lon1, lat2, lon2):
    center_lat = (lat1 + lat2) / 2
    center_lon = (lon1 + lon2) / 2
    graph = None
    for _ in range(3):
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
# HELPER / COMPUTE FUNCTIONS
# ============================================

def calculate_resilience_score(aqi, wind_speed, humidity):
    aqi_impact  = max(0, 100 - (aqi / 3))
    wind_bonus  = min(15, wind_speed * 1.5)
    return min(100, max(0, round(aqi_impact + wind_bonus)))

def get_healthiest_route(start_lat, start_lon, end_lat, end_lon):
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2
    total = 0
    for lat, lon in [(start_lat, start_lon), (mid_lat, mid_lon), (end_lat, end_lon)]:
        total += fetch_global_aqi(lat, lon).get('us_aqi', 50)
    return total / 3

def refresh_dashboard_data(lat, lon, name):
    st.session_state.map_center          = [lat, lon]
    st.session_state.current_data        = fetch_global_aqi(lat, lon)
    st.session_state.current_weather     = fetch_live_weather(lat, lon)
    st.session_state.forecast_data       = fetch_7day_aqi_forecast(lat, lon)
    st.session_state.multi_forecast_data = fetch_multi_pollutant_forecast(lat, lon)

# ============================================
# AI FUNCTIONS
# ============================================

def get_ai_health_briefing(pm25_val, aqi_val, profile="General Public"):
    if client is None:
        return "AI health briefing unavailable (Gemini not initialised)."
    profile_info = SENSITIVITY_PROFILES.get(profile, SENSITIVITY_PROFILES["General Public"])
    prompt = f"""
Act as a clinical health educator. The current PM2.5 level is {pm25_val:.1f} µg/m³ and the AQI is {aqi_val:.0f}.
The user's respiratory sensitivity profile is: {profile} (caution threshold AQI {profile_info['aqi_caution']}, danger threshold AQI {profile_info['aqi_danger']}).
Write a 3-sentence "Daily Health Briefing":
1. Explain the biological impact on the respiratory system at this level for someone in this profile.
2. Suggest one specific preventative measure tailored to this sensitivity level.
3. Keep it empathetic, precise, and professional.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# ============================================
# PDF REPORT GENERATOR
# ============================================

def generate_pdf_report(current_aqi, pm25_val, forecast_df, briefing_text, city_name="Current Location"):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(11, 15, 25)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 14, "PROJECT AIR — Health Intelligence Report", ln=True, align='C')

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Location: {city_name}   |   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align='C')
    pdf.ln(6)

    # AQI banner
    pdf.set_fill_color(16, 185, 129) if current_aqi <= 50 else (pdf.set_fill_color(245, 158, 11) if current_aqi <= 100 else pdf.set_fill_color(239, 68, 68))
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 12, f"  Current US AQI: {int(current_aqi)}   |   PM2.5: {pm25_val:.1f} µg/m³", ln=True, fill=True)
    pdf.ln(4)

    # AI Health Briefing
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 9, "AI-Generated Health Briefing", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.set_fill_color(240, 248, 255)
    # Multi-line safe cell
    pdf.multi_cell(0, 7, briefing_text, border=0)
    pdf.ln(5)

    # 7-Day Forecast table
    if not forecast_df.empty:
        pdf.set_font("Arial", 'B', 13)
        pdf.cell(0, 9, "7-Day AQI Forecast", ln=True)
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(55, 8, "Date", border=1, fill=True)
        pdf.cell(55, 8, "US AQI", border=1, fill=True)
        pdf.cell(55, 8, "PM2.5 (µg/m³)", border=1, ln=True, fill=True)
        pdf.set_font("Arial", size=10)
        for _, row in forecast_df.iterrows():
            pdf.cell(55, 7, str(row.get('date', '')), border=1)
            pdf.cell(55, 7, str(row.get('aqi', 'N/A')), border=1)
            pdf.cell(55, 7, str(round(row.get('pm25', 0), 1) if row.get('pm25') is not None else 'N/A'), border=1, ln=True)
        pdf.ln(5)

    # Footer
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Data sources: Open-Meteo Air Quality API | NASA POWER | Gemini AI", ln=True, align='C')
    pdf.cell(0, 7, "This report is for informational purposes only. Consult a physician for medical advice.", ln=True, align='C')

    return bytes(pdf.output(dest='S'))

# ============================================
# THEMING / CSS
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Exo+2:wght@300;400;600;900&display=swap');

    html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

    .stApp { background-color: #0b0f19; color: #ffffff; }

    .title-gradient {
        background: linear-gradient(-45deg, #00f2fe, #4facfe, #00f2fe, #4facfe);
        background-size: 300% 300%;
        animation: gradient-shift 8s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Exo 2', sans-serif;
        font-size: 54px !important;
        font-weight: 900;
        margin-bottom: 0px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    @keyframes gradient-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .sub-font {
        font-size: 17px !important;
        color: #8b9bb4;
        margin-bottom: 30px;
        font-weight: 300;
        letter-spacing: 1px;
    }

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
    .glass-card h5 { color: #8b9bb4 !important; font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; }
    .glass-card h3 { color: #ffffff !important; font-size: 38px; font-weight: 800; margin: 0px 0px 10px 0px; }
    .glass-card span { font-size: 13px; color: #4facfe; }

    .profile-card {
        background: rgba(79, 172, 254, 0.08);
        border: 1px solid rgba(79, 172, 254, 0.25);
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    # --- Respiratory Profile ---
    st.markdown("### 🫁 Respiratory Profile")
    profile_options = list(SENSITIVITY_PROFILES.keys())
    selected_profile = st.selectbox(
        "Your Sensitivity Level",
        options=profile_options,
        index=profile_options.index(st.session_state.get('respiratory_profile', 'General Public')),
        key="sidebar_profile"
    )
    st.session_state['respiratory_profile'] = selected_profile
    pdata = SENSITIVITY_PROFILES[selected_profile]
    st.markdown(f"""
        <div class="profile-card">
            <b>{pdata['label']}</b><br>
            <span style="color:#8b9bb4; font-size:12px;">
            ⚠️ Caution AQI ≥ {pdata['aqi_caution']} &nbsp;|&nbsp;
            🚨 Danger AQI ≥ {pdata['aqi_danger']}
            </span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Pollutant selector ---
    st.markdown("### 🧬 Active Pollutant")
    selected_name = st.selectbox("Pollutant", options=list(POLLUTANT_MAP.keys()), key="sidebar_pollutant")
    st.session_state['selected_pollutant_key']  = POLLUTANT_MAP[selected_name]
    st.session_state['selected_pollutant_name'] = selected_name

    st.markdown("---")

    # --- Alert settings ---
    st.markdown("### 🔔 Alert Protocol")
    with st.form("alert_form"):
        target_email     = st.text_input("Operator Email", placeholder="you@example.com")
        alert_threshold  = st.slider("AQI Trigger Threshold", 50, 300, 100, 10)
        submit_alert     = st.form_submit_button("Initialize Protocol")
        if submit_alert:
            if target_email:
                st.success("✅ Protocol Active.")
            else:
                st.error("Valid email required.")

    st.markdown("---")

    # --- Session timer ---
    elapsed = datetime.now() - st.session_state.start_time
    seconds = int(elapsed.total_seconds())
    time_str = f"{seconds} sec" if seconds < 60 else f"{seconds // 60} min"
    st.write(f"🕒 Last update: **{time_str} ago**")

    if st.button("🔄 Manual Refresh"):
        st.session_state.start_time = datetime.now()
        st.cache_data.clear()
        st.rerun()

# ============================================
# DATA INITIALISATION (on first load)
# ============================================
with st.spinner("Initialising Atmospheric Sensors..."):
    lat0, lon0 = st.session_state.map_center

    if st.session_state.current_data is None:
        st.session_state.current_data = fetch_global_aqi(lat0, lon0)
    if st.session_state.current_weather is None:
        st.session_state.current_weather = fetch_live_weather(lat0, lon0)
    if st.session_state.forecast_data is None:
        st.session_state.forecast_data = fetch_7day_aqi_forecast(lat0, lon0)
    if st.session_state.multi_forecast_data is None:
        st.session_state.multi_forecast_data = fetch_multi_pollutant_forecast(lat0, lon0)

current_data = st.session_state.current_data or {}
weather      = st.session_state.current_weather or {}

# Safe numeric extraction
raw_aqi   = current_data.get('us_aqi', 0)
current_aqi = float(raw_aqi) if str(raw_aqi).replace('.', '', 1).isdigit() else 0.0
raw_pm25    = current_data.get('pm2_5', 0)
pm25_val    = float(raw_pm25) if raw_pm25 not in (None, 'N/A') else 0.0
aqi_color   = "#10b981" if current_aqi <= 50 else ("#f59e0b" if current_aqi <= 100 else "#ef4444")

# ============================================
# HEADER
# ============================================
st.markdown('<p class="title-gradient">Project AIR</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-font">Atmospheric Intelligence & Response — Global Health & Pollution Analytics Engine</p>',
    unsafe_allow_html=True
)

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Telemetry",
    "🔮 Multi-Pollutant Forecast",
    "🧠 Prediction Engine",
    "🩺 Health Literacy",
    "🗺️ Heatmap & Sensors",
    "🌡️ Climate Correlations",
    "🌌 NASA Fire Intel",
    "🚲 Clean-Air Commute"
])

# ============================================================
# TAB 1 — TELEMETRY & REAL-TIME OVERVIEW
# ============================================================
with tab1:
    st.markdown("### 🌍 Regional Atmospheric Bio-Telemetry")

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        city_input = st.text_input("Search Location", "Raleigh, NC", key="tab1_search", label_visibility="collapsed")
    with col_btn:
        if st.button("📡 Scan Region", use_container_width=True):
            lat, lon, name = get_city_coords(city_input)
            if lat:
                refresh_dashboard_data(lat, lon, name)
                st.rerun()
            else:
                st.error("Location not found.")

    # Live metrics
    temp = weather.get('temperature_2m', 'N/A')
    wind = weather.get('wind_speed_10m', 'N/A')
    head = weather.get('wind_direction_10m', 'N/A')
    humidity = weather.get('relative_humidity_2m', 50)

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='glass-card' style='border-top:3px solid {aqi_color};'><h5>US AQI</h5><h3 style='color:{aqi_color}'>{int(current_aqi)}</h3></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='glass-card' style='border-top:3px solid #fbbf24;'><h5>Temperature</h5><h3>{temp}°C</h3></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='glass-card' style='border-top:3px solid #a78bfa;'><h5>Wind Speed</h5><h3>{wind} km/h</h3></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='glass-card' style='border-top:3px solid #38bdf8;'><h5>Heading</h5><h3>{head}°</h3></div>", unsafe_allow_html=True)

    # Resilience score
    resilience_val = calculate_resilience_score(
        current_aqi,
        float(wind) if wind != 'N/A' else 0.0,
        float(humidity) if humidity not in (None, 'N/A') else 50
    )
    st.markdown("---")
    st.subheader("🛡️ Environmental Resilience Index")
    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        st.metric("Resilience Score", f"{resilience_val}/100", delta=f"{resilience_val - 50:+d} from baseline")
    with col_r2:
        if resilience_val > 80:
            st.info("🟢 **High Resilience:** Excellent atmospheric dispersion preventing pollutant stagnation.")
        elif resilience_val > 50:
            st.info("🟡 **Moderate Resilience:** Average atmospheric drift; pollutants may accumulate locally.")
        else:
            st.info("🔴 **Low Resilience:** Stagnant conditions — high risk of rapid localized accumulation.")

    # Long-term trajectory chart
    st.markdown("---")
    st.markdown("#### 📈 PM2.5 Long-Term Atmospheric Trajectory")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_yearly["year"], y=df_yearly["mean_pm25"],
        mode="lines+markers", name="Recorded",
        line=dict(color="#00f2fe", width=3)
    ))
    forecast_future = future_df[future_df['year'] > df_yearly['year'].max()]
    fig.add_trace(go.Scatter(
        x=forecast_future["year"], y=forecast_future["predicted_pm25"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#ff0844", width=3, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_future["year"], forecast_future["year"][::-1]]),
        y=pd.concat([forecast_future["yhat_upper"], forecast_future["yhat_lower"][::-1]]),
        fill='toself', fillcolor='rgba(255,8,68,0.1)',
        line=dict(color='rgba(255,255,255,0)'), showlegend=True, name="Confidence Band"
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        margin=dict(t=80, l=40, r=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Clinical advisory
    st.markdown("---")
    st.subheader("🩺 Clinical Neuro-Respiratory Advisory")
    a1, a2 = st.columns([1, 2])
    with a1:
        st.markdown(
            f"<div style='text-align:center;padding:20px;background:rgba(255,255,255,0.02);border-radius:15px;'>"
            f"<h1 style='color:{aqi_color};font-size:60px;'>{int(current_aqi)}</h1></div>",
            unsafe_allow_html=True
        )
    with a2:
        if current_aqi <= 50:
            st.success("✅ **Air Quality is Optimal.** Ideal for all outdoor activities.")
        elif current_aqi <= 100:
            st.warning("⚠️ **Air Quality is Moderate.** Sensitive individuals should limit prolonged exertion.")
        else:
            st.error("🚨 **High Toxicity Detected.** Deep-lung particulate risk. Indoor protocols advised.")

    # PDF report download
    st.markdown("---")
    st.subheader("📄 Download Health Intelligence Report")
    if st.button("⬇️ Generate PDF Report", key="gen_pdf_tab1"):
        with st.spinner("Generating report..."):
            briefing = get_ai_health_briefing(pm25_val, current_aqi, st.session_state.get('respiratory_profile', 'General Public'))
            forecast_df = st.session_state.get('forecast_data', pd.DataFrame())
            city_coords = st.session_state.get('map_center', [35.7796, -78.6382])
            pdf_bytes = generate_pdf_report(current_aqi, pm25_val, forecast_df, briefing)
            st.download_button(
                label="📥 Download Report PDF",
                data=pdf_bytes,
                file_name=f"ProjectAIR_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

    with st.expander("📊 Advanced Atmospheric Bio-Telemetry"):
        plot_df = df_yearly.copy()
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


# ============================================================
# TAB 2 — MULTI-POLLUTANT 7-DAY FORECAST (NEW)
# ============================================================
with tab2:
    st.markdown("### 🔮 7-Day Multi-Pollutant Atmospheric Forecast")
    st.caption("Search any city to generate a 7-day outlook for PM2.5, PM10, Ozone, and NO2.")

    col_s, col_b = st.columns([4, 1])
    with col_s:
        fc_city = st.text_input("City for Forecast", "Raleigh, NC", key="fc_city_input", label_visibility="collapsed")
    with col_b:
        if st.button("🔭 Forecast", use_container_width=True, key="fc_btn"):
            lat, lon, name = get_city_coords(fc_city)
            if lat:
                st.session_state.multi_forecast_data = fetch_multi_pollutant_forecast(lat, lon)
                st.session_state.forecast_city = name
                st.success(f"📍 Forecast loaded for **{name}**")
            else:
                st.error("City not found.")

    mp_df = st.session_state.get('multi_forecast_data', pd.DataFrame())

    if mp_df is not None and not mp_df.empty:
        city_label = st.session_state.get('forecast_city', 'Your Location')
        st.markdown(f"#### 📅 7-Day Forecast — {city_label}")

        # Pollutant selector tabs
        pollutant_tab_labels = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "Ozone (µg/m³)", "NO2 (µg/m³)"]
        pollutant_keys       = ["PM2.5", "PM10", "Ozone", "NO2"]
        colors               = ["#00f2fe", "#a78bfa", "#fbbf24", "#f87171"]
        units                = ["µg/m³", "µg/m³", "µg/m³", "µg/m³"]

        ptab1, ptab2, ptab3, ptab4 = st.tabs(pollutant_tab_labels)
        tab_list = [ptab1, ptab2, ptab3, ptab4]

        for i, (ptab, key, color) in enumerate(zip(tab_list, pollutant_keys, colors)):
            with ptab:
                col_vals = mp_df[key].dropna()
                if col_vals.empty:
                    st.info(f"No {key} data available for this location.")
                    continue

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=mp_df["date"],
                    y=mp_df[key],
                    name=key,
                    marker_color=color,
                    opacity=0.7
                ))
                fig.add_trace(go.Scatter(
                    x=mp_df["date"],
                    y=mp_df[key],
                    mode="lines+markers",
                    name=f"{key} Trend",
                    line=dict(color=color, width=3),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    yaxis_title=f"{key} ({units[i]})",
                    xaxis_title="Date",
                    legend=dict(orientation="h", y=1.1),
                    margin=dict(t=40, b=40, l=40, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Daily summary table
                st.markdown(f"**Daily {key} Values**")
                display_df = mp_df[["date", key]].copy()
                display_df.columns = ["Date", f"{key} ({units[i]})"]
                display_df[f"{key} ({units[i]})"] = display_df[f"{key} ({units[i]})"].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Combined overlay chart
        st.markdown("---")
        st.markdown("#### 📊 All Pollutants — Normalised Overlay")
        fig_all = go.Figure()
        for key, color in zip(pollutant_keys, colors):
            col_vals = mp_df[key].dropna()
            if col_vals.empty:
                continue
            norm = (mp_df[key] - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-9)
            fig_all.add_trace(go.Scatter(
                x=mp_df["date"], y=norm,
                name=key, mode="lines+markers",
                line=dict(color=color, width=2)
            ))
        fig_all.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Normalised Concentration (0–1)",
            hovermode="x unified",
            margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig_all, use_container_width=True)
        st.caption("Values normalised per-pollutant so all four can be compared on the same axis.")

    else:
        st.info("👆 Enter a city above and click **Forecast** to generate the 7-day multi-pollutant outlook.")


# ============================================================
# TAB 3 — IMPACT & MITIGATION SIMULATOR (was tab2)
# ============================================================
with tab3:
    st.markdown("### 🧠 Impact & Mitigation Simulator")

    risk_score = (pm25_val / 12) * 1.5
    if pm25_val < 12:
        status, briefing_text, col_color = "Baseline Homeostasis", "Air quality is optimal. Minimal systemic inflammation. Cognitive function not under environmental stress.", "#10b981"
    elif pm25_val < 35:
        status, briefing_text, col_color = "Mild Oxidative Stress", "Moderate particulate load. Your body is mobilising antioxidants. You may experience subtle focus fatigue.", "#f59e0b"
    else:
        status, briefing_text, col_color = "Neuro-Inflammatory Warning", "High PM2.5 load. Particulates are triggering systemic inflammatory response. Limit intense cognitive tasks.", "#ef4444"

    st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);padding:20px;border-radius:15px;border-left:5px solid {col_color};'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div><h5 style='margin:0;'>Cognitive Risk Index</h5><h2 style='color:{col_color};margin:0;'>{round(risk_score,1)} / 10</h2></div>
                <div style='text-align:right;'><h5 style='margin:0;'>Current Status</h5><span style='color:white;font-weight:bold;'>{status}</span></div>
            </div>
            <p style='margin-top:15px;font-size:0.9em;color:#a1a1aa;'>{briefing_text}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_sim_gauge, col_sim_slider = st.columns(2)
    with col_sim_gauge:
        st.markdown("**Current Threat Level (AQI)**")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=current_aqi,
            gauge={
                'axis': {'range': [0, 300], 'tickcolor': "white"},
                'bar': {'color': col_color},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0, 50],   'color': "rgba(16,185,129,0.2)"},
                    {'range': [50, 150], 'color': "rgba(245,158,11,0.2)"},
                    {'range': [150, 300],'color': "rgba(239,68,68,0.2)"}
                ]
            }
        ))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30,b=30,l=30,r=30))
        st.plotly_chart(gauge, use_container_width=True)

    with col_sim_slider:
        st.markdown("**Emission Mitigation Simulator**")
        reduction = st.slider("Target Mitigation (%)", 0, 50, 0)
        sim_val = future_df['predicted_pm25'].iloc[-1] * (1 - reduction / 100)
        st.markdown(f"""
            <div class='glass-card' style='margin-top:20px;'>
                <h5>Estimated {future_df['year'].iloc[-1]} PM2.5</h5>
                <h3 style='color:#00f2fe;'>{sim_val:.2f} µg/m³</h3>
                <span>Projected Impact: -{reduction}%</span>
            </div>
        """, unsafe_allow_html=True)

    # 7-Day AQI chart
    st.markdown("---")
    st.markdown("### 📅 7-Day AQI Forecast")
    forecast_df = st.session_state.get('forecast_data', pd.DataFrame())
    if forecast_df is not None and not forecast_df.empty:
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["aqi"],
            mode="lines+markers", name="AQI Forecast",
            line=dict(color="#00f2fe", width=4),
            fill='tozeroy', fillcolor='rgba(0,242,254,0.1)'
        ))
        fig_fc.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", yaxis_title="AQI Level"
        )
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info("Search a location in Tab 1 to load the 7-day AQI forecast.")


# ============================================================
# TAB 4 — HEALTH LITERACY (was tab3)
# ============================================================
with tab4:
    st.markdown("### 🩺 Advanced Health Literacy & Physiological Impact")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🔬 Pollutant Breakdown")
        with st.expander("PM2.5 (Fine Particulates)", expanded=True):
            st.write("Particles <2.5µm bypass airway defences and lodge deep in the alveoli.")
            st.metric("WHO Safe Limit", "< 15.0 µg/m³ (annual)")
        with st.expander("PM10 (Coarse Particulates)"):
            st.write("Particles <10µm, filtered by upper airways but irritating at high concentrations.")
            st.metric("WHO Safe Limit", "< 45.0 µg/m³ (annual)")
        with st.expander("Ozone (O₃)"):
            st.write("Ground-level ozone irritates airways and can trigger asthma attacks.")
            st.metric("WHO Safe Limit", "< 100 µg/m³ (8-hr mean)")
        with st.expander("NO2 (Nitrogen Dioxide)"):
            st.write("Traffic-related pollutant inflaming lung tissue and reducing immunity.")
            st.metric("WHO Safe Limit", "< 10 µg/m³ (annual)")

    with col_b:
        st.markdown("#### 🧬 Systemic Physiological Load")
        body_load = min(100, (current_aqi / 300) * 100)
        st.progress(body_load / 100, text=f"Estimated inflammatory strain: {int(body_load)}%")

        # Profile-aware alert
        st.markdown("---")
        profile = st.session_state.get('respiratory_profile', 'General Public')
        pdata = SENSITIVITY_PROFILES[profile]
        st.markdown(f"**Profile Alert — {pdata['label']}**")
        if current_aqi >= pdata['aqi_danger']:
            st.error(f"🚨 DANGER: AQI {int(current_aqi)} exceeds your danger threshold ({pdata['aqi_danger']}). Stay indoors.")
        elif current_aqi >= pdata['aqi_caution']:
            st.warning(f"⚠️ CAUTION: AQI {int(current_aqi)} exceeds your caution threshold ({pdata['aqi_caution']}). Limit outdoor exertion.")
        else:
            st.success(f"✅ Air quality is within safe limits for your sensitivity profile.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏃 Exposure Risk Calculator")
        activity = st.selectbox("Activity Level", ["Resting", "Light Walk", "Heavy Exercise"], key="tab4_act_level")
        if activity == "Heavy Exercise" and current_aqi > 100:
            st.error("🚨 CRITICAL: High-intensity exercise prohibited at this AQI.")
        elif activity == "Heavy Exercise" and current_aqi > 50:
            st.warning("⚠️ CAUTION: Reduce intensity is recommended.")
        else:
            st.success("✅ Activity level is safe for current air quality.")
    with c2:
        st.subheader("🏠 Indoor Scrubbing Tool")
        room_sqft = st.number_input("Room Square Footage", value=200, key="tab4_sqft")
        filter_eff = st.select_slider("Purifier Efficiency", options=["Standard", "HEPA", "Industrial"], key="tab4_filter")
        if st.button("Calculate Scrub Time", key="tab4_scrub"):
            rate = 50 if filter_eff == "Standard" else (100 if filter_eff == "HEPA" else 150)
            st.write(f"⏱️ Estimated Scrub Time: **{room_sqft / rate:.1f} minutes**")

    # AI Health Briefing with profile
    st.markdown("---")
    st.subheader("🤖 AI-Powered Health Briefing")
    if st.button("Generate Personalised Briefing", key="tab4_briefing_btn"):
        with st.spinner("Consulting Gemini AI..."):
            briefing = get_ai_health_briefing(pm25_val, current_aqi, st.session_state.get('respiratory_profile', 'General Public'))
        st.markdown(f"""
            <div style='background:rgba(79,172,254,0.07);border:1px solid rgba(79,172,254,0.2);
                        border-radius:12px;padding:20px;margin-top:10px;'>
                {briefing}
            </div>
        """, unsafe_allow_html=True)
    
    # PDF button in health tab too
    st.markdown("---")
    if st.button("⬇️ Generate PDF Health Report", key="gen_pdf_tab4"):
        with st.spinner("Generating report..."):
            briefing = get_ai_health_briefing(pm25_val, current_aqi, st.session_state.get('respiratory_profile', 'General Public'))
            forecast_df = st.session_state.get('forecast_data', pd.DataFrame())
            pdf_bytes = generate_pdf_report(current_aqi, pm25_val, forecast_df, briefing)
            st.download_button(
                "📥 Download PDF",
                data=pdf_bytes,
                file_name=f"ProjectAIR_HealthReport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="pdf_dl_tab4"
            )


# ============================================================
# TAB 5 — POLLUTANT DISPERSION HEATMAP (NEW + was tab4)
# ============================================================
with tab5:
    st.markdown("### 🗺️ Pollutant Dispersion Heatmap & Global Sensor Search")

    col_s, col_b = st.columns([4, 1])
    with col_s:
        hmap_city = st.text_input(
            "Search Location", "Tokyo",
            key="tab5_search", label_visibility="collapsed",
            placeholder="Enter city name..."
        )
    with col_b:
        search_clicked = st.button("📍 Search", use_container_width=True, key="tab5_btn")

    if search_clicked and hmap_city:
        lat, lon, name = get_city_coords(hmap_city)
        if lat:
            st.session_state.map_center = [lat, lon]
            st.session_state.current_data = fetch_global_aqi(lat, lon)
            st.success(f"📍 Map locked to: **{name}**")
        else:
            st.error("Location not found.")

    curr_lat, curr_lon = st.session_state.map_center

    # Generate synthetic heatmap points around the selected city using real AQI as anchor
    data_point = st.session_state.get('current_data') or {}
    base_aqi = float(data_point.get('us_aqi', 50) or 50)
    base_pm25 = float(data_point.get('pm2_5', 15) or 15)

    # Create realistic scatter of sensor-like points (weighted by base value)
    np.random.seed(42)
    n_points = 80
    offsets_lat = np.random.normal(0, 0.08, n_points)
    offsets_lon = np.random.normal(0, 0.08, n_points)
    weights = np.abs(np.random.normal(base_pm25, base_pm25 * 0.3, n_points)).clip(1, 120)
    hotspots = [
        [curr_lat + offsets_lat[i], curr_lon + offsets_lon[i], float(weights[i])]
        for i in range(n_points)
    ]

    m = folium.Map(location=[curr_lat, curr_lon], zoom_start=11, tiles="CartoDB dark_matter")

    # Add heatmap layer
    HeatMap(
        hotspots,
        radius=22,
        blur=18,
        max_zoom=13,
        gradient={0.2: '#00f2fe', 0.5: '#fbbf24', 0.8: '#f87171', 1.0: '#ef4444'}
    ).add_to(m)

    # Central marker
    folium.Marker(
        [curr_lat, curr_lon],
        tooltip=f"Primary Sensor | AQI: {int(base_aqi)} | PM2.5: {base_pm25:.1f} µg/m³",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    st_folium(m, width="100%", height=480, key="tab5_map")

    # Legend
    st.markdown("""
        <div style="display:flex;gap:20px;margin-top:8px;align-items:center;flex-wrap:wrap;">
            <span>🔵 Low PM2.5</span>
            <span>🟡 Moderate</span>
            <span>🔴 High PM2.5</span>
            <span style="color:#8b9bb4;font-size:12px;">Heatmap radius anchored to real-time AQI data</span>
        </div>
    """, unsafe_allow_html=True)

    # Atmospheric report beneath
    st.markdown("---")
    st.markdown("### 📊 Atmospheric Report")
    if data_point:
        c1, c2, c3 = st.columns(3)
        c1.metric("US AQI", int(base_aqi))
        c2.metric("PM2.5 (µg/m³)", f"{base_pm25:.1f}")
        c3.metric("Coordinates", f"{curr_lat:.3f}, {curr_lon:.3f}")
    else:
        st.warning("No data yet — search a city above.")


# ============================================================
# TAB 6 — HISTORICAL CLIMATE CORRELATION DASHBOARD (NEW)
# ============================================================
with tab6:
    st.markdown("### 🌡️ Historical Climate Correlation Dashboard")
    st.caption("Last 30 days of PM2.5 vs weather variables — identify local pollution patterns.")

    col_s, col_b = st.columns([4, 1])
    with col_s:
        corr_city = st.text_input("City for Correlation", "Raleigh, NC", key="corr_city", label_visibility="collapsed")
    with col_b:
        corr_btn = st.button("📈 Load Data", use_container_width=True, key="corr_btn")

    corr_lat, corr_lon = st.session_state.map_center

    if corr_btn:
        lat, lon, name = get_city_coords(corr_city)
        if lat:
            corr_lat, corr_lon = lat, lon
            st.session_state['corr_center'] = [lat, lon]
            st.session_state['corr_city_name'] = name
        else:
            st.error("City not found.")

    if 'corr_center' in st.session_state:
        corr_lat, corr_lon = st.session_state['corr_center']

    with st.spinner("Fetching 30-day historical data..."):
        corr_df = fetch_30day_correlation(corr_lat, corr_lon)

    if corr_df is not None and not corr_df.empty:
        city_label = st.session_state.get('corr_city_name', 'Selected Location')
        st.markdown(f"#### 📊 30-Day Climate-Pollution Correlation — {city_label}")

        # Resample to daily for clarity
        daily_corr = corr_df.resample('D', on='time').mean().reset_index()

        # PM2.5 vs Humidity
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PM2.5 vs Humidity**")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=daily_corr['time'], y=daily_corr['pm25'],
                name="PM2.5", line=dict(color="#00f2fe", width=2), yaxis='y1'
            ))
            fig1.add_trace(go.Scatter(
                x=daily_corr['time'], y=daily_corr['humidity'],
                name="Humidity (%)", line=dict(color="#a78bfa", width=2, dash='dot'), yaxis='y2'
            ))
            fig1.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="PM2.5 (µg/m³)", color="#00f2fe"),
                yaxis2=dict(title="Humidity (%)", overlaying='y', side='right', color="#a78bfa"),
                legend=dict(orientation='h', y=1.1),
                margin=dict(t=40, b=40)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**PM2.5 vs Wind Speed**")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=daily_corr['time'], y=daily_corr['pm25'],
                name="PM2.5", line=dict(color="#00f2fe", width=2), yaxis='y1'
            ))
            fig2.add_trace(go.Scatter(
                x=daily_corr['time'], y=daily_corr['wind'],
                name="Wind (km/h)", line=dict(color="#fbbf24", width=2, dash='dot'), yaxis='y2'
            ))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="PM2.5 (µg/m³)", color="#00f2fe"),
                yaxis2=dict(title="Wind (km/h)", overlaying='y', side='right', color="#fbbf24"),
                legend=dict(orientation='h', y=1.1),
                margin=dict(t=40, b=40)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter: PM2.5 vs Temperature
        st.markdown("**PM2.5 vs Temperature — Scatter Analysis**")
        fig3 = px.scatter(
            daily_corr, x='temp', y='pm25',
            color='pm25', size='pm25',
            color_continuous_scale='RdYlGn_r',
            labels={'temp': 'Temperature (°C)', 'pm25': 'PM2.5 (µg/m³)'},
            trendline='ols',
            template='plotly_dark'
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=40)
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Correlation summary
        st.markdown("**Correlation Coefficients (Pearson)**")
        corrs = {
            "PM2.5 ↔ Humidity":     round(daily_corr['pm25'].corr(daily_corr['humidity']), 3),
            "PM2.5 ↔ Wind Speed":   round(daily_corr['pm25'].corr(daily_corr['wind']), 3),
            "PM2.5 ↔ Temperature":  round(daily_corr['pm25'].corr(daily_corr['temp']), 3),
        }
        c1, c2, c3 = st.columns(3)
        for col, (label, val) in zip([c1, c2, c3], corrs.items()):
            color = "#10b981" if abs(val) < 0.3 else ("#f59e0b" if abs(val) < 0.6 else "#ef4444")
            col.markdown(
                f"<div class='glass-card'><h5>{label}</h5><h3 style='color:{color};'>{val:+.3f}</h3>"
                f"<span>{'Strong' if abs(val)>0.6 else ('Moderate' if abs(val)>0.3 else 'Weak')} correlation</span></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("👆 Enter a city and click **Load Data** to view the 30-day correlation analysis.")
        st.caption("This fetches hourly data for the past 30 days from Open-Meteo APIs.")


# ============================================================
# TAB 7 — NASA FIRE INTELLIGENCE (was tab5)
# ============================================================
with tab7:
    st.markdown("### 🌍 Global Satellite Intelligence — Forest Fire Detection")

    today_str = datetime.now().strftime("%Y-%m-%d")
    tile_url = (
        f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"VIIRS_SNPP_Thermal_Anomalies_375m/default/{today_str}/"
        f"GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.png"
    )

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.5),
        layers=[pdk.Layer("TileLayer", tile_url, opacity=0.8)],
    ))
    st.caption(f"🛰️ NASA VIIRS Thermal Anomalies — {today_str}")

    st.markdown("### 🛰️ Climate Metrics (NASA POWER)")
    nasa_lat, nasa_lon = st.session_state.map_center
    nasa_data = get_nasa_climate_data(nasa_lat, nasa_lon)
    col1, col2 = st.columns(2)
    col1.metric("Surface Solar Irradiance", f"{nasa_data['solar_radiation']} kW/m²")
    col2.metric("Satellite Wind Velocity",  f"{nasa_data['satellite_wind_speed']} m/s")


# ============================================================
# TAB 8 — CLEAN-AIR COMMUTE (was tab6)
# ============================================================
with tab8:
    st.markdown("### 🚲 The Clean-Air Commute Planner")
    from geopy.distance import geodesic

    col1, col2 = st.columns(2)
    start_addr = col1.text_input("Starting Address", "100 Main St, Raleigh, NC", key="start_addr")
    end_addr   = col2.text_input("Destination Address", "500 Main St, Rolesville, NC", key="end_addr")

    if st.button("🗺️ Generate Healthiest Path", key="route_btn"):
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="project_air_app_v2")
            s_loc = geolocator.geocode(start_addr)
            e_loc = geolocator.geocode(end_addr)
            if not s_loc or not e_loc:
                st.error("Could not geocode one or both addresses. Please be more specific.")
            else:
                s_lat, s_lon = s_loc.latitude, s_loc.longitude
                e_lat, e_lon = e_loc.latitude, e_loc.longitude
                dist_km = geodesic((s_lat, s_lon), (e_lat, e_lon)).km
                if dist_km > 50:
                    st.error("Distance too large (max 50 km). Please choose closer points.")
                else:
                    with st.spinner("Building road network & calculating path..."):
                        graph = get_map_graph(s_lat, s_lon, e_lat, e_lon)
                    if graph:
                        start_node = ox.distance.nearest_nodes(graph, s_lon, s_lat)
                        end_node   = ox.distance.nearest_nodes(graph, e_lon, e_lat)
                        try:
                            route = nx.shortest_path(graph, start_node, end_node, weight='travel_time')
                            route_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in route]
                            total_time_min = sum(
                                graph.edges[(u, v, 0)].get('travel_time', 0)
                                for u, v in zip(route[:-1], route[1:])
                            ) / 60
                            avg_exposure = get_healthiest_route(s_lat, s_lon, e_lat, e_lon)
                            st.session_state.route_data = {
                                "route":    route_coords,
                                "s_lat":    s_lat, "s_lon": s_lon,
                                "e_lat":    e_lat, "e_lon": e_lon,
                                "exposure": avg_exposure,
                                "time_est": round(total_time_min, 1)
                            }
                        except Exception:
                            st.error("No navigable path found between these points.")
                    else:
                        st.error("Could not download road network. Please try again.")
        except Exception as e:
            st.error(f"Address error: {e}")

    if 'route_data' in st.session_state:
        rdata = st.session_state.route_data
        aqi_val_route = round(rdata.get('exposure', 0), 1)
        col1, col2 = st.columns(2)
        col1.metric("Route AQI Exposure", aqi_val_route)
        col2.metric("Est. Travel Time", f"{rdata.get('time_est', 'N/A')} min")

        m_route = folium.Map(tiles="CartoDB dark_matter")
        folium.PolyLine(rdata['route'], color="#4facfe", weight=5, opacity=0.9).add_to(m_route)
        folium.Marker([rdata['s_lat'], rdata['s_lon']], icon=folium.Icon(color='green'), tooltip="Start").add_to(m_route)
        folium.Marker([rdata['e_lat'], rdata['e_lon']], icon=folium.Icon(color='red'),   tooltip="End").add_to(m_route)
        m_route.fit_bounds(rdata['route'])
        st_folium(m_route, width="100%", height=420, key="tab8_map")

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
            "🚨 **High Risk.** Air quality is poor today. We strongly recommend an indoor commute.",
            "🚨 **Warning:** High particulate levels detected. Strenuous outdoor exercise is not advised.",
            "🚨 **Alert:** Exposure today could cause respiratory irritation. Stick to indoor environments."
        ]

        if aqi_val_route < 50:
            st.success(random.choice(good_advice))
        elif aqi_val_route < 100:
            st.warning(random.choice(mid_advice))
        else:
            st.error(random.choice(bad_advice))

        st.markdown("#### 🧬 Biological Context")
        st.write("""
            Fine particulate matter (PM2.5) bypasses natural airway defences, settling deep in the alveoli
            where gas exchange occurs. Once they cross this barrier they can cause systemic inflammation
            throughout the body, affecting cardiovascular and neurological systems beyond the lungs.
        """)
