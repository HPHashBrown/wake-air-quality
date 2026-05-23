import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# Refresh every 5 minutes (300,000 ms)
count = st_autorefresh(interval=300000, key="datarefresh")

# Initialize session state for the timer if it doesn't exist
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

FIRMS_API_KEY = "5ced48a900256b1fac376db945c3980d" 

def get_nasa_climate_data(lat, lon):
    # Calculate a date 30 days ago to ensure data is available
    target_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    
    # URL using a guaranteed historical date
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,WS2M&community=RE&longitude={lon}&latitude={lat}&start={target_date}&end={target_date}&format=JSON"
    
    try:
        response = requests.get(url).json()
        # Access the dictionary safely
        data = response['properties']['parameter']
        solar_values = list(data['ALLSKY_SFC_SW_DWN'].values())
        wind_values = list(data['WS2M'].values())
        
        # Only return if we found actual data, not -999
        solar = solar_values[0] if solar_values[0] != -999 else "N/A"
        wind = wind_values[0] if wind_values[0] != -999 else "N/A"
        
        return {"solar_radiation": solar, "satellite_wind_speed": wind}
    except Exception as e:
        return {"solar_radiation": "N/A", "satellite_wind_speed": "N/A"}

def fetch_wildfire_data():
    # NASA FIRMS Global Fire Data (latest 24h)
    url = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_placeholder/map/C6/firms/csv/USA_contiguous_and_Hawaii_24h.csv"
    try:
        df = pd.read_csv(url)
        # Filter for fires roughly in the Eastern US to keep the list relevant
        # (Latitude 30-40, Longitude -85 to -75 covers the general NC region)
        nc_fires = df[(df['latitude'] > 30) & (df['latitude'] < 40) & 
                      (df['longitude'] > -85) & (df['longitude'] < -75)]
        return nc_fires
    except:
        return pd.DataFrame()


try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


metric_key = "pm2_5" 
selected_metric = "PM2.5" 
selected_risks = ["Good (0-50)", "Moderate (51-100)", "Unhealthy (101+)"]



def fetch_wildfire_data():
    # Use the specific FIRMS key
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_API_KEY}/VIIRS_SNPP_NRT/USA/1"
    
    try:
        # Attempt to read the data
        df = pd.read_csv(url)
        
        # Standardize columns (NASA sometimes changes names)
        # We rename whatever the latitude/longitude column is called to 'latitude'/'longitude'
        if 'latitude' not in df.columns and 'lat' in df.columns:
            df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
            
        # Filter for NC area
        nc_fires = df[(df['latitude'] >= 34) & (df['latitude'] <= 37) & 
                      (df['longitude'] >= -84) & (df['longitude'] <= -75)]
        return nc_fires
        
    except Exception as e:
        # If anything goes wrong, this block catches the error 
        # so your app doesn't crash.
        return pd.DataFrame()


# ============================================
# PAGE CONFIG & HIGH-TECH THEMING
# ============================================
st.set_page_config(page_title="Project Wake AQI", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.set_page_config(page_title="Project Wake AQI", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Animated Global Background */
    .stApp { background-color: #0b0f19; color: #ffffff; }
    
    /* Animated Gradient Title */
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
    
    /* Glassmorphism UI */
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

    /* Sidebar Styling */
    .css-1d391kg { background-color: #0b0f19; border-right: 1px solid rgba(255, 255, 255, 0.08); }
    </style>
""", unsafe_allow_html=True)

# ============================================
# FEATURE 4: ALERT NOTIFICATION SYSTEM (SIDEBAR)
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

with st.sidebar:
    st.markdown("---")
    
    # Calculate time passed since page load
    elapsed = datetime.now() - st.session_state.start_time
    seconds = int(elapsed.total_seconds())
    
    if seconds < 60:
        time_str = f"{seconds} seconds"
    else:
        minutes = seconds // 60
        time_str = f"{minutes} minutes"
        
    st.write(f"🕒 Data updated: **{time_str} ago**")

with st.sidebar:
    st.markdown("---")
    
    # 1. The Manual Reset Button
    if st.button("🔄 Manual Refresh"):
        # Reset the timer to now
        st.session_state.start_time = datetime.now()
        
        # Pro-Tip: If you want to force the data to re-fetch 
        # (clearing the cache for current session), uncomment the line below:
        # st.cache_data.clear() 
        
        st.rerun() # Forces the app to re-run immediately

    # 2. The Timer Display
    elapsed = datetime.now() - st.session_state.start_time
    seconds = int(elapsed.total_seconds())
    
    if seconds < 60:
        time_str = f"{seconds} seconds"
    else:
        minutes = seconds // 60
        time_str = f"{minutes} minutes"
        
    st.write(f"🕒 Data updated: **{time_str} ago**")



# ============================================
# DATA FETCHING (API)
# ============================================
@st.cache_data(ttl=3600)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    try:
        return requests.get(url).json().get("current", None)
    except: return None

@st.cache_data(ttl=3600)
def fetch_current_aqi(lat=35.7796, lon=-78.6382, metric_key="pm2_5"):
    # We inject the metric_key into the API string dynamically
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,{metric_key}"
    try: 
        return requests.get(url).json().get("current", {})
    except: return {}

# ============================================
# PROCESSING & MODELING
# ============================================
# Automatically use uploaded file if exists, otherwise fallback to local CSV
if uploaded_file is not None:
    df_yearly = pd.read_csv(uploaded_file).sort_values("year")
else:
    try:
        df_yearly = pd.read_csv("wake_pm25_by_year.csv").sort_values("year")
    except:
        df_yearly = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022, 2023], "mean_pm25": [10.2, 9.8, 8.5, 9.2, 8.1, 7.9]})

current_year = datetime.now().year
with st.spinner("Initializing Atmospheric Sensors & Predictive AI..."):
    current_aqi_data = fetch_current_aqi()
    live_weather = fetch_live_weather(35.7796, -78.6382)

# ============================================
# FEATURE 1: ADVANCED AI FORECASTING (PROPHET)
# ============================================
if PROPHET_AVAILABLE:
    # Prepare data for Prophet
    prophet_df = df_yearly.copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df['year'], format='%Y')
    prophet_df = prophet_df.rename(columns={'mean_pm25': 'y'})
    
    # Initialize and train Meta's Prophet
    m = Prophet(yearly_seasonality=True)
    m.fit(prophet_df)
    
    # Predict 10 years into the future
    future = m.make_future_dataframe(periods=10, freq='YS')
    forecast = m.predict(future)
    
    # Extract prediction data
    future_df = pd.DataFrame({
        "year": forecast['ds'].dt.year,
        "predicted_pm25": forecast['yhat'],
        "yhat_lower": forecast['yhat_lower'],
        "yhat_upper": forecast['yhat_upper']
    })
else:
    # Fallback to standard Linear Regression if Prophet isn't installed
    X = df_yearly[["year"]]
    y = df_yearly["mean_pm25"]
    model = LinearRegression().fit(X, y)
    future_years = np.arange(current_year, current_year + 11)
    future_preds = model.predict(future_years.reshape(-1, 1))
    future_df = pd.DataFrame({
        "year": future_years, 
        "predicted_pm25": future_preds,
        "yhat_lower": future_preds - 1.5, # Mock confidence bounds
        "yhat_upper": future_preds + 1.5
    })

current_aqi = current_aqi_data.get('us_aqi', 0)

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

# ============================================
# UI LAYOUT
# ============================================
# ============================================
# UI LAYOUT
# ============================================
st.markdown('<p class="title-gradient">Project Wake AQI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Statewide Atmospheric PM2.5 Analytics Engine.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Telemetry & Forecasting", "🧠 Predictive Scenario Core", "🩺 Health Literacy", "🛰️ Statewide Vector Map", "🌌NASA Space Intelligence"])

# --- TAB 1: OVERVIEW & ADVANCED CHART ---
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

# --- TAB 2: ADVANCED ANALYTICS ---
with tab2:
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Current Threat Level (AQI)**")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=current_aqi, gauge={'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': "white"}, 'bar': {'color': "#00f2fe"}, 'bgcolor': "rgba(255,255,255,0.05)", 'steps': [{'range': [0, 50], 'color': "rgba(16, 185, 129, 0.3)"}, {'range': [50, 100], 'color': "rgba(245, 158, 11, 0.3)"}, {'range': [100, 300], 'color': "rgba(239, 68, 68, 0.3)"}]}))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(gauge, use_container_width=True)
    with g2:
        st.markdown("**Emission Mitigation Simulator**")
        reduction = st.slider("Simulated Reduction (%)", 0, 50, 0)
        sim_val = future_df['predicted_pm25'].iloc[-1] * (1 - (reduction/100))
        st.metric(f"Estimated {future_df['year'].iloc[-1]} PM2.5", f"{sim_val:.2f} µg/m³", delta=f"-{reduction}% impact")

# --- TAB 3: HEALTH LITERACY ---
with tab3:
    st.markdown("### 🩺 Biological Impact Protocols")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### The Regional Vector")
        st.write("Urban expansion across North Carolina impacts air quality. Monitoring particulate matter is critical.")
    with c2:
        st.markdown("#### Medical Glossary")
        with st.expander("🔬 What is PM2.5?"):
            st.write("Particulate matter 2.5 micrometers or smaller, capable of entering the bloodstream.")
        with st.expander("🛡️ What is a Healthy Level?"):
            st.write("The EPA considers annual mean concentrations below 12.0 µg/m³ as sustainable for the general public.")

# --- TAB 4: NC MAP ---
with tab4:
    st.markdown("### 🛰️ North Carolina Sensor Mesh")
        
    # --- GLOBAL DATA FETCHING ---
# Now takes dynamic lat/lon
@st.cache_data(ttl=3600)
def fetch_global_aqi(lat, lon, metric="pm2_5"):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,{metric}"
    try: 
        response = requests.get(url).json()
        return response.get("current", {})
    except: 
        return {}

# --- IN YOUR MAIN TAB 4 (GLOBAL MAP) ---
with tab4:
    st.markdown("### 🌍 Global Exploration")
    
    # 1. Initialize Map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
    m.add_child(folium.LatLngPopup()) # Allows clicking to see coordinates
    
    # 2. Render Map and capture clicks
    map_data = st_folium(m, width=1000, height=500)
    
    # 3. Dynamic Data Logic
    if map_data['last_clicked']:
        lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
        st.info(f"📍 Analyzing coordinates: {lat:.2f}, {lon:.2f}")
        
        # Fetch data for clicked location
        data = fetch_global_aqi(lat, lon, pollutant)
        if data:
            st.metric(f"AQI at {lat:.2f}, {lon:.2f}", data.get('us_aqi', 'N/A'))
    else:
        st.write("Click anywhere on the map to analyze local air quality.")
            
            # Determine the category string for filtering
        if aqi <= 50: category = "Good (0-50)"; color = "#10b981"
            elif aqi <= 100: category = "Moderate (51-100)"; color = "#f59e0b"
            else: category = "Unhealthy (101+)"; color = "#ef4444"
            
            # Only render if the category is checked in the sidebar filter
            if category in selected_risks:
                folium.CircleMarker(
                    location=coords, 
                    radius=12, 
                    color=color, 
                    fill=True, 
                    fill_color=color, 
                    fill_opacity=0.8,
                    popup=f"<b>{city}</b><br><b>{selected_metric}:</b> {val}<br><b>AQI:</b> {aqi}"
                ).add_to(m)
        
        st_folium(m, use_container_width=True, height=600)
    except Exception as e:
        st.error(f"Orbital Feed Interrupted: {e}")

fires = fetch_wildfire_data()


    # Update your tabs definition:
# tab1, tab2, tab3, tab4, tab5 = st.tabs([... , "🛰️ Space Intelligence"])

with tab5:
    st.markdown("### 🌍 Satellite-Derived Context")
    st.write("Cross-referencing local nodes with NASA Earth Observation platforms.")
    
    nasa_data = get_nasa_climate_data(35.7796, -78.6382)
    
    col1, col2 = st.columns(2)
    col1.metric("Surface Solar Irradiance", f"{nasa_data['solar_radiation']} kW/m²", help="NASA POWER: Measures potential for ozone formation.")
    col2.metric("Satellite Wind Velocity", f"{nasa_data['satellite_wind_speed']} m/s", help="NASA POWER: High-altitude wind drift data.")
    
    st.info("💡 **Why this matters:** NASA monitors surface solar radiation because high levels contribute to ground-level ozone formation, which directly impacts your AQI readings.")

with tab5:
    st.markdown("### 🔥 Real-Time Wildfire Hotspots")
    fires = fetch_wildfire_data()
    
    if not fires.empty:
        st.warning(f"Detected {len(fires)} active fire hotspots in the Eastern US.")
        st.dataframe(fires[['latitude', 'longitude', 'acq_time', 'bright_ti4']])
    else:
        st.success("No active fire hotspots detected in the immediate NC region.")


# --- PLACE THIS AFTER THE 4 COLUMNS IN TAB 1 ---
    st.markdown("---")
    st.subheader("🩺 Public Health Advisory")
    
    # Dynamic Alert Logic
    if current_aqi <= 50:
        st.success("✅ **Air Quality is Good.** Air quality is considered satisfactory, and air pollution poses little or no risk.")
    elif current_aqi <= 100:
        st.warning("⚠️ **Air Quality is Moderate.** Air quality is acceptable; however, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.")
    elif current_aqi <= 150:
        st.error("🚫 **Unhealthy for Sensitive Groups.** Members of sensitive groups may experience health effects. The general public is less likely to be affected.")
    else:
        st.error("🚨 **Health Alert.** Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects. Reduce prolonged outdoor exertion.")
