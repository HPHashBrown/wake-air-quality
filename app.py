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
# Refresh every 5 minutes (300,000 ms)
count = st_autorefresh(interval=300000, key="datarefresh")

# Initialize session state for the map center
if 'map_center' not in st.session_state:
    st.session_state.map_center = [35.7796, -78.6382] # Default to Raleigh

# Initialize session state for the timer if it doesn't exist
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

@st.cache_data(ttl=3600)
def fetch_global_aqi(lat, lon, metric):
    # This URL must use the 'metric' variable to update the request
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,{metric}"
    try: 
        response = requests.get(url).json()
        return response.get("current", {})
    except: 
        return {}

FIRMS_API_KEY = "5ced48a900256b1fac376db945c3980d" 
metric_key = "pm2_5" 
selected_metric = "PM2.5" 
selected_risks = ["Good (0-50)", "Moderate (51-100)", "Unhealthy (101+)"]

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# ============================================
# API FUNCTIONS (MOVED TO TOP TO PREVENT ERRORS)
# ============================================
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

@st.cache_data(ttl=3600)
def fetch_global_aqi(lat, lon, metric="pm2_5"):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,{metric}"
    try: 
        response = requests.get(url).json()
        return response.get("current", {})
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
    user_data = fetch_current_aqi(user_lat, user_lon)
    control_data = fetch_current_aqi(control_lat, control_lon)

    user_aqi = user_data.get('us_aqi', 0)
    control_aqi = control_data.get('us_aqi', 0)

    diff = user_aqi - control_aqi

    if diff > 15:
        return f"🚨 Micro-climate Alert: Your location is {diff} AQI points dirtier than the nearby {control_name} node."
    elif diff < -5:
        return "✅ You are currently in a high-quality air pocket."
    return "Air quality is consistent across your local area."

# ============================================
# PAGE CONFIG & HIGH-TECH THEMING
# ============================================
st.set_page_config(page_title="Wake AQI", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

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
    st.markdown("### 📂 Data Ingest")
    uploaded_file = st.file_uploader("Upload custom CSV", type=['csv'])

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

current_year = datetime.now().year 

if uploaded_file is not None:
    df_yearly = pd.read_csv(uploaded_file).sort_values("year")
else:
    try:
        df_yearly = pd.read_csv("wake_pm25_by_year.csv").sort_values("year")
    except:
        df_yearly = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022, 2023], "mean_pm25": [10.2, 9.8, 8.5, 9.2, 8.1, 7.9]})

# Replace the current_aqi_data call in your PROCESSING section:
with st.spinner("Initializing Atmospheric Sensors..."):
    # Use the selected_key from your Sidebar
    live_weather = fetch_live_weather(35.7796, -78.6382)

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

current_aqi = current_pollutant_data.get('us_aqi', 0)

# ============================================
# MAIN UI LAYOUT
# ============================================
st.markdown('<p class="title-gradient">Project Wake AQI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Statewide Atmospheric PM2.5 Analytics Engine.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Telemetry & Forecasting", "🧠 Predictive Scenario Core", "🩺 Health Literacy", "🛰️ Statewide Vector Map", "🌌NASA Space Intelligence"])

# --- TAB 1: OVERVIEW ---
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='glass-card'><h5>US AQI</h5><h3>{current_aqi}</h3></div>", unsafe_allow_html=True)
    with m2:
        val = f"{live_weather['temperature_2m']}°C" if live_weather else "N/A"
        st.markdown(f"<div class='glass-card'><h5>Thermal State</h5><h3>{val}</h3><span>Raleigh Node</span></div>", unsafe_allow_html=True)
    with m3:
        val = f"{live_weather['wind_speed_10m']} km/h" if live_weather else "N/A"
        st.markdown(f"<div class='glass-card'><h5>Wind Velocity</h5><h3>{val}</h3><span>Dispersion Rate</span></div>", unsafe_allow_html=True)
    with m4:
        val = f"{live_weather['wind_direction_10m']}°" if live_weather else "N/A"
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

# --- TAB 2: ANALYTICS ---
with tab2:
    st.markdown("### 🛠️ Impact & Mitigation Simulator")
    
    # --- 1. HEALTH LITERACY & COGNITIVE ENGINE ---
    # Ensure we handle the potential 'N/A' from the API safely
    raw_pm25 = current_data.get('pm2_5', 0)
    pm25_val = float(raw_pm25) if raw_pm25 != 'N/A' else 0.0
    
    # Cognitive Risk Logic
    risk_score = (pm25_val / 12) * 1.5
    
    col_a, col_b = st.columns([1, 2])
    col_a.metric("Cognitive Risk Index", f"{round(risk_score, 1)}/10")
    
    # Clinical Translator logic
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

    # --- 2. EXISTING EMISSION MITIGATION SIMULATOR ---
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Current Threat Level (AQI)**")
        # Ensure current_aqi exists or default to 0
        current_aqi = current_data.get('us_aqi', 0)
        if current_aqi == 'N/A': current_aqi = 0
        
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=float(current_aqi), 
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
        # Using the forecast model from your previous setup
        sim_val = future_df['predicted_pm25'].iloc[-1] * (1 - (reduction/100))
        st.metric(f"Estimated {future_df['year'].iloc[-1]} PM2.5", f"{sim_val:.2f} µg/m³", delta=f"-{reduction}% impact")

# --- TAB 3: HEALTH ---

with tab3:
    st.markdown("### 🩺Health Literacy")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.info("### The Scale of the Invisible")
        st.write("A human hair is 50μm. PM2.5 is <2.5μm. It is roughly **1/30th the width of a hair**.")
        # Visualizing the scale helps understand why filters fail.


    with col2:
        st.warning("### The Systemic Journey")
        st.write("1. **Deep Lung Penetration:** Reaches the alveoli.")
        st.write("2. **Bloodstream Entry:** Enters systemic circulation.")
        st.write("3. **Chronic Inflammation:** Triggers long-term oxidative stress.")


    st.markdown("---")

    # --- DYNAMIC SAFE EXPOSURE CALCULATOR ---
    st.markdown("### ⏳ Your Daily 'Safe Exposure' Budget")

    if current_aqi > 0:
        # Budget logic: If AQI is 100, you have ~8 hours of 'safe' outdoor activity before impact.
        # This is a heuristic model: 800 / AQI = Safe hours remaining
        safe_hours = max(0, 800 / (current_aqi * 1.5)) 

        st.metric("Estimated Safe Hours Left Today", f"{safe_hours:.1f} Hours")

        if safe_hours > 6:
            st.success("✅ **Air Quality is Clear.** No restrictions on your outdoor exposure.")
        elif safe_hours > 3:
            st.warning("⚠️ **Caution.** Limit intense outdoor exercise. Your inflammatory budget is depleting.")
        else:
            st.error("🚫 **Alert.** Your inflammatory budget is low. Indoor air protocol recommended.")
    else:
        st.write("Fetching sensor data to calculate your budget...")


# --- TAB 4: MAP / SEARCH ---
# --- TAB 4: MAP / SEARCH ---
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

    m = folium.Map(location=st.session_state.map_center, zoom_start=8, tiles="CartoDB dark_matter")
    m.add_child(folium.LatLngPopup())
    folium.Marker(st.session_state.map_center, tooltip="Sensor Hub").add_to(m)

    map_data = st_folium(m, width="100%", height=400, key="map_view")

    if map_data and map_data.get('last_clicked'):
        new_lat = map_data['last_clicked']['lat']
        new_lon = map_data['last_clicked']['lng']
        st.session_state.map_center = [new_lat, new_lon]
        st.rerun()

    # Everything below here MUST have exactly 4 spaces of indentation
# ... inside your Tab 4 block ...
    st.markdown("### 📊 Atmospheric Report")
    curr_lat, curr_lon = st.session_state.map_center

    # Use the selection from the Sidebar via session_state
    active_pollutant_key = st.session_state.get('selected_pollutant_key', 'pm2_5')
    active_pollutant_name = st.session_state.get('selected_pollutant_name', 'PM2.5')
    
    try:
        data = fetch_global_aqi(curr_lat, curr_lon, st.session_state.get('selected_pollutant_key', 'pm2_5'))
        aqi = data.get('us_aqi', 'N/A')
        # Pass the dynamic key to the fetch function
        data = fetch_global_aqi(curr_lat, curr_lon, active_pollutant_key)
        
        # Get the value for the selected pollutant dynamically
        pollutant_value = data.get(active_pollutant_key, 'N/A')
        aqi_val = data.get('us_aqi', 'N/A')
        
        c1, c2, c3 = st.columns(3)
        c1.metric("US AQI", f"{aqi_val}")
        # Dynamically label the metric with the selected pollutant name
        c2.metric(f"{active_pollutant_name}", f"{pollutant_value}")
        c3.metric("Node", f"{curr_lat:.2f}, {curr_lon:.2f}")

        if aqi != 'N/A':
            c1, c2 = st.columns(2)
            c1.metric("US AQI Index", f"{aqi}")
            c2.metric("Coordinate Node", f"{curr_lat:.2f}, {curr_lon:.2f}")
        else:
            st.warning("Sensor data currently unavailable for this coordinate.")
    except Exception as e:
        st.error(f"Error retrieving sensor data: {e}")
# --- TAB 5: SPACE INTEL ---
with tab5:
    st.markdown("### 🌍 Satellite-Derived Context")
    st.write("Cross-referencing local nodes with NASA Earth Observation platforms.")

    nasa_data = get_nasa_climate_data(35.7796, -78.6382)
    col1, col2 = st.columns(2)
    col1.metric("Surface Solar Irradiance", f"{nasa_data['solar_radiation']} kW/m²", help="NASA POWER: Measures potential for ozone formation.")
    col2.metric("Satellite Wind Velocity", f"{nasa_data['satellite_wind_speed']} m/s", help="NASA POWER: High-altitude wind drift data.")
    st.info("💡 **Why this matters:** NASA monitors surface solar radiation because high levels contribute to ground-level ozone formation, which directly impacts your AQI readings.")

    st.markdown("### 🔥 Real-Time Wildfire Hotspots")
    fires = fetch_wildfire_data()
    if not fires.empty:
        st.warning(f"Detected {len(fires)} active fire hotspots in the Eastern US.")
        st.dataframe(fires[['latitude', 'longitude', 'acq_time', 'bright_ti4']])
    else:
        st.success("No active fire hotspots detected.")
