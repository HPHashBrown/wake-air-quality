import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ============================================
# PAGE CONFIGURATION & CUSTOM CSS
# ============================================
st.set_page_config(page_title="Wake AQI & Weather Forecast", page_icon="🌤️", layout="wide")

st.markdown("""
    <style>
    .big-font {font-size: 46px !important; font-weight: 700; color: #1E3A8A; margin-bottom: 0px;}
    .sub-font {font-size: 18px !important; color: #64748B; margin-bottom: 25px;}
    .metric-card {background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0;}
    .good-badge {background-color: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 5px; font-weight: bold; font-size: 14px;}
    .mod-badge {background-color: #fef08a; color: #854d0e; padding: 4px 8px; border-radius: 5px; font-weight: bold; font-size: 14px;}
    .unh-badge {background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 5px; font-weight: bold; font-size: 14px;}
    </style>
""", unsafe_allow_html=True)

# ============================================
# SECRETS MANAGEMENT
# ============================================
EMAIL = st.secrets.get("EPA_EMAIL", "test@example.com")
API_KEY = st.secrets.get("EPA_API_KEY", "testkey")

# ============================================
# LIVE EXTERNAL DATA INTEGRATIONS
# ============================================
@st.cache_data(ttl=3600)  # Cache weather for 1 hour to keep it perfectly live
def fetch_live_weather():
    # Coordinates centered on Wake County / Raleigh area
    lat, lon = 35.7796, -78.6382
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    try:
        res = requests.get(url).json()
        return res.get("current", None)
    except:
        return None

@st.cache_data(ttl=86400)  # Cache EPA historical data for 24 hours
def fetch_pm25(year):
    url = (f"https://aqs.epa.gov/data/api/dailyData/byCounty?"
           f"email={EMAIL}&key={API_KEY}&param=88101&bdate={year}0101&edate={year}1231"
           f"&state=37&county=183")
    try:
        res = requests.get(url).json()
        if "Data" not in res: return None
        df = pd.DataFrame(res["Data"])
        return df[["date_local", "arithmetic_mean"]] if not df.empty else None
    except:
        return None

def get_aqi_badge(value):
    if value <= 12.0: return "<span class='good-badge'>🟢 Good</span>"
    elif value <= 35.4: return "<span class='mod-badge'>🟡 Moderate</span>"
    else: return "<span class='unh-badge'>🔴 Unhealthy</span>"

# ============================================
# DATA PIPELINE
# ============================================
try:
    df_yearly = pd.read_csv("wake_pm25_by_year.csv").sort_values("year")
except:
    df_yearly = pd.DataFrame(columns=["year", "mean_pm25"])

current_year = datetime.now().year

with st.spinner("Syncing data feeds..."):
    df_daily = fetch_pm25(current_year)
    live_weather = fetch_live_weather()

if df_daily is not None and not df_yearly.empty:
    mean_pm25 = df_daily["arithmetic_mean"].mean()
    new_row = pd.DataFrame({"year": [current_year], "mean_pm25": [mean_pm25]})
    df_yearly = df_yearly[df_yearly["year"] != current_year]
    df_yearly = pd.concat([df_yearly, new_row], ignore_index=True).sort_values("year")

if not df_yearly.empty:
    X = df_yearly[["year"]]
    y = df_yearly["mean_pm25"]
    model = LinearRegression().fit(X, y)
    
    future_years = np.arange(current_year + 1, current_year + 11)
    future_preds = model.predict(future_years.reshape(-1, 1))
    future_df = pd.DataFrame({"year": future_years, "predicted_pm25": future_preds})

# ============================================
# APP LAYOUT
# ============================================
st.markdown('<p class="big-font">Wake County Air Quality Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Predictive analytics and live environmental tracking metrics.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & Forecast", 
    "🗃️ Data Explorer", 
    "🩺 Health Literacy", 
    "🧠 Advanced Analytics"
])

# --------------------------------------------
# TAB 1: OVERVIEW & FORECAST (With Live Weather Sidebar/Metrics)
# --------------------------------------------
with tab1:
    # Top Row: Environmental Context
    st.subheader("🌐 Current Conditions")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        if not df_yearly.empty:
            st.markdown(f"<div class='metric-card'><h5>{current_year} PM2.5 Average</h5><h3>{y.iloc[-1]:.2f} µg/m³</h3>{get_aqi_badge(y.iloc[-1])}</div>", unsafe_allow_html=True)
    
    # Integrating the live weather data points directly into the layout
    with m_col2:
        if live_weather:
            st.markdown(f"<div class='metric-card'><h5>Live Local Temp</h5><h3>{live_weather['temperature_2m']}°C</h3><span>Real-time tracking</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='metric-card'><h5>Live Local Temp</h5><h3>--</h3><span>Feed offline</span></div>", unsafe_allow_html=True)
            
    with m_col3:
        if live_weather:
            st.markdown(f"<div class='metric-card'><h5>Relative Humidity</h5><h3>{live_weather['relative_humidity_2m']}%</h3><span>Updated hourly</span></div>", unsafe_allow_html=True)
            
    with m_col4:
        if live_weather:
            st.markdown(f"<div class='metric-card'><h5>Wind Speed</h5><h3>{live_weather['wind_speed_10m']} km/h</h3><span>Affects particle dispersion</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical and Forecasting Trends
    if not df_yearly.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], mode="markers+lines", name="Historical Avg", line=dict(color="#1E3A8A", width=3)))
        fig.add_trace(go.Scatter(x=df_yearly["year"], y=model.predict(X), mode="lines", name="Linear Trendline", line=dict(color="gray", width=1.5, dash="dash")))
        fig.add_trace(go.Scatter(x=future_df["year"], y=future_df["predicted_pm25"], mode="markers+lines", name="10-Year Prediction", line=dict(color="#EF4444", width=2.5, dash="dot")))
        
        fig.update_layout(title="Wake County PM2.5 Tracking & Multi-Year Projection", xaxis_title="Year", yaxis_title="PM2.5 (µg/m³)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# TAB 4: ADVANCED ANALYTICS (Gauge & Simulator)
# --------------------------------------------
with tab4:
    st.subheader("Deep-Dive Predictive Analytics")
    
    if not df_yearly.empty:
        col_gauge, col_sim = st.columns(2)
        
        with col_gauge:
            st.markdown("**Current PM2.5 Scale Evaluation**")
            gauge_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = y.iloc[-1],
                gauge = {
                    'axis': {'range': [0, 40]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 12], 'color': "#dcfce7"},
                        {'range': [12, 35.4], 'color': "#fef08a"},
                        {'range': [35.4, 40], 'color': "#fee2e2"}
                    ]
                }
            ))
            gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)
            
        with col_sim:
            st.markdown("**Interactive Scenario Simulator**")
            reduction = st.slider("Simulate Emissions Abatement (%)", 0, 50, 0, 5)
            
            if reduction > 0:
                sim_preds = future_preds * (1 - (reduction / 100))
                sim_fig = go.Figure()
                sim_fig.add_trace(go.Scatter(x=future_df["year"], y=future_preds, mode="lines", name="Baseline", line=dict(color="gray", dash="dash")))
                sim_fig.add_trace(go.Scatter(x=future_df["year"], y=sim_preds, mode="lines+markers", name="Adjusted Scenario", line=dict(color="green")))
                sim_fig.update_layout(height=240, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white")
                st.plotly_chart(sim_fig, use_container_width=True)

# [Remaining data display codes belong to tab2 and tab3 as structured previously]
