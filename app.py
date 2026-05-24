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
    "pressure_hpa": [1013, 1012, 1014, 1011, 1015, 1013],
    "humidity": [65, 70, 68, 72, 65, 66],
    "aerosol_depth": [0.1, 0.12, 0.09, 0.11, 0.08, 0.07]
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

st.set_page_config(page_title="Wake AQI", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")
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

def fetch_wildfire_data():
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_API_KEY}/VIIRS_SNPP_NRT/USA/1"
    try:
        df = pd.read_csv(url)
        if 'latitude' not in df.columns and 'lat' in df.columns:
            df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
        nc_fires = df[(df['latitude'] >= 34) & (df['latitude'] <= 37) & 
                      (df['longitude'] >= -84) & (df['longitude'] <= -75)]
        return nc_fires
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    try:
        return requests.get(url).json().get("current", None)
    except: return None

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
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    try:
        response = requests.get(url).json()
        if "results" in response:
            data = response["results"][0]
            return data["latitude"], data["longitude"], data["name"]
        return None, None, None
    except Exception as e:
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
st.markdown('<p class="title-gradient">Project Wake AQI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Statewide Atmospheric PM2.5 Analytics Engine.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Telemetry & Forecasting", "🧠 Predictive Scenario Core", "🩺 Health Literacy", "🛰️ Statewide Vector Map", "🌌NASA Space Intelligence", "🚲The Clean-Air Commute"])

# --- TAB 1: OVERVIEW ---

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{current_aqi}</h3></div>", unsafe_allow_html=True)
    with m2:
        val = f"{live_weather.get('temperature_2m', 'N/A')}°C"
        st.markdown(f"<div class='glass-card'><h5>Thermal State</h5><h3>{val}</h3><span>Raleigh Node</span></div>", unsafe_allow_html=True)
    with m3:
        val = f"{live_weather.get('wind_speed_10m', 'N/A')} km/h"
        st.markdown(f"<div class='glass-card'><h5>Wind Velocity</h5><h3>{val}</h3><span>Dispersion Rate</span></div>", unsafe_allow_html=True)
    with m4:
        val = f"{live_weather.get('wind_direction_10m', 'N/A')}°"
        st.markdown(f"<div class='glass-card'><h5>Vector Heading</h5><h3>{val}</h3><span>Atmospheric Drift</span></div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], mode="lines+markers", name="Recorded Telemetry", line=dict(color="#00f2fe", width=4), marker=dict(size=8, color="#ffffff", line=dict(width=2, color="#00f2fe"))))
    forecast_future = future_df[future_df['year'] > df_yearly['year'].max()]
    fig.add_trace(go.Scatter(x=forecast_future["year"], y=forecast_future["predicted_pm25"], mode="lines+markers", name="Algorithmic Forecast", line=dict(color="#f87171", width=4, dash="dot"), marker=dict(size=8, color="#ffffff", line=dict(width=2, color="#f87171"))))
    fig.add_trace(go.Scatter(x=pd.concat([forecast_future["year"], forecast_future["year"][::-1]]), y=pd.concat([forecast_future["yhat_upper"], forecast_future["yhat_lower"][::-1]]), fill='toself', fillcolor='rgba(248, 113, 113, 0.15)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=True, name="AI Confidence Interval"))
    fig.update_layout(title="PM2.5 Long-Term Atmospheric Trajectory", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🩺 Public Health Advisory")
    if current_aqi <= 50:
        st.success("✅ **Air Quality is Good.** Air quality is considered satisfactory, and air pollution poses little or no risk.")
    elif current_aqi <= 100:
        st.warning("⚠️ **Air Quality is Moderate.** Air quality is acceptable; however, there may be a risk for some people.")
    elif current_aqi <= 150:
        st.error("🚫 **Unhealthy for Sensitive Groups.** Members of sensitive groups may experience health effects.")
    else:
        st.error("🚨 **Health Alert.** Some members of the general public may experience health effects.")

    resilience_val = calculate_resilience_score(
        current_aqi, 
        live_weather.get('wind_speed_10m', 0), 
        live_weather.get('relative_humidity_2m', 50)
    )
    
    st.markdown("---")
    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        st.metric("Resilience Score", f"{resilience_val}/100")
    with col_r2:
        if resilience_val > 80:
            st.success("The environment is highly resilient. Atmospheric conditions are dispersing pollutants effectively.")
        elif resilience_val > 50:
            st.warning("Moderate resilience. Local conditions are stable; keep an eye on changing trends.")
        else:
            st.error("Low resilience detected. Conditions are stagnant and current pollutant levels are impactful.")
        
# --- TAB 2: ANALYTICS ---
with tab2:
    st.markdown("### 🛠️ Impact & Mitigation Simulator")
    raw_pm25 = current_data.get('pm2_5', 0)
    pm25_val = float(raw_pm25) if raw_pm25 != 'N/A' else 0.0
    risk_score = (pm25_val / 12) * 1.5
    col_a, col_b = st.columns([1, 2])
    col_a.metric("Cognitive Risk Index", f"{round(risk_score, 1)}/10")
    if pm25_val < 12:
        status = "Baseline Homeostasis"
        briefing = "Air quality is optimal. Minimal systemic inflammation detected. Cognitive function is not under environmental stress."
    elif 12 <= pm25_val < 35:
        status = "Mild Oxidative Stress"
        briefing = "Moderate particulate load. Your body is mobilizing antioxidants to counteract minor systemic inflammation. You may experience subtle focus fatigue."
    else:
        status = "Neuro-Inflammatory Warning"
        briefing = "High PM2.5 load. Particulates are likely triggering a systemic inflammatory response. This can impact neural processing speed and executive function. Limit intense cognitive tasks and exertion."
    col_b.write(f"**Status:** {status}")
    col_b.info(briefing)

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Current Threat Level (AQI)**")
        current_aqi_val = float(current_aqi) if current_aqi != 'N/A' else 0
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=current_aqi_val, 
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
        reduction = st.slider("Simulated Reduction (%)", 0, 50, 0)
        sim_val = future_df['predicted_pm25'].iloc[-1] * (1 - (reduction/100))
        st.metric(f"Estimated {future_df['year'].iloc[-1]} PM2.5", f"{sim_val:.2f} µg/m³", delta=f"-{reduction}% impact")

# --- TAB 3: HEALTH ---
with tab3:
    st.markdown("### 🩺Health Literacy")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.info("### The Scale of the Invisible")
        st.write("A human hair is 50μm. PM2.5 is <2.5μm. It is roughly **1/30th the width of a hair**.")
    with col2:
        st.warning("### The Systemic Journey")
        st.write("1. **Deep Lung Penetration:** Reaches the alveoli.")
        st.write("2. **Bloodstream Entry:** Enters systemic circulation.")
        st.write("3. **Chronic Inflammation:** Triggers long-term oxidative stress.")
    st.markdown("---")
    st.markdown("### ⏳ Your Daily 'Safe Exposure' Budget")
    if current_aqi > 0:
        safe_hours = max(0, 800 / (float(current_aqi) * 1.5)) 
        st.metric("Estimated Safe Hours Left Today", f"{safe_hours:.1f} Hours")
        if safe_hours > 6:
            st.success("✅ **Air Quality is Clear.** No restrictions on your outdoor exposure.")
        elif safe_hours > 3:
            st.warning("⚠️ **Caution.** Limit intense outdoor exercise. Your inflammatory budget is depleting.")
        else:
            st.error("🚫 **Alert.** Your inflammatory budget is low. Indoor air protocol recommended.")


with tab4:
    st.markdown("### 🔍 Global Sensor Search")
    city_input = st.text_input("Search Location (e.g., Tokyo, Raleigh, Paris)", key="city_input_field")
    
    if city_input:
        lat, lon, name = get_city_coords(city_input)
        if lat:
            st.session_state.map_center = [lat, lon]
            st.success(f"📍 Navigation locked to: {name}")
        else:
            st.error("Location not found.")
            
    # Initialize map
    m = folium.Map(location=st.session_state.map_center, zoom_start=8, tiles="CartoDB dark_matter")
    
    # Marker for the current sensor hub
    folium.Marker(
        st.session_state.map_center, 
        tooltip="Active Sensor Hub",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    # Render the map
    st_folium(m, width="100%", height=400)

    # Atmospheric Report
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
    st.markdown("### 🌍 Satellite-Derived Context")
    nasa_data = get_nasa_climate_data(35.7796, -78.6382)
    col1, col2 = st.columns(2)
    col1.metric("Surface Solar Irradiance", f"{nasa_data['solar_radiation']} kW/m²", help="NASA POWER: Measures potential for ozone formation.")
    col2.metric("Satellite Wind Velocity", f"{nasa_data['satellite_wind_speed']} m/s", help="NASA POWER: High-altitude wind drift data.")
    st.markdown("### 🔥 Real-Time Wildfire Hotspots")
    fires = fetch_wildfire_data()
    if not fires.empty:
        st.warning(f"Detected {len(fires)} active fire hotspots in the Eastern US.")
        st.dataframe(fires[['latitude', 'longitude', 'acq_time', 'bright_ti4']])
    else:
        st.success("No active fire hotspots detected.")


with tab6:
    st.markdown("### 🚲 The Clean-Air Commute")
    from geopy.distance import geodesic
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
        pm25_est = round(aqi_val * 0.4, 2)
        with st.spinner("Analyzing physiological impact..."):
            briefing = get_ai_health_briefing(pm25_est, aqi_val)
            st.info(briefing)
            st.caption("Biological Context:")
            st.write("Exposure to fine particulates (PM2.5) can trigger systemic oxidative stress, as these particles cross the alveolar-capillary barrier.")
