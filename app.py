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
st.set_page_config(page_title="Wake AQI Forecast", page_icon="🌤️", layout="wide")

# Custom CSS for UI upgrades
st.markdown("""
    <style>
    .big-font {font-size: 50px !important; font-weight: 700; color: #1E3A8A;}
    .sub-font {font-size: 20px !important; color: #64748B; margin-bottom: 30px;}
    .metric-card {background-color: #F8FAFC; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
    .good-badge {background-color: #dcfce7; color: #166534; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .mod-badge {background-color: #fef08a; color: #854d0e; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .unh-badge {background-color: #fee2e2; color: #991b1b; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ============================================
# SECRETS MANAGEMENT
# ============================================
EMAIL = st.secrets.get("EPA_EMAIL", "test@example.com")
API_KEY = st.secrets.get("EPA_API_KEY", "testkey")

# ============================================
# HELPER FUNCTIONS
# ============================================
@st.cache_data(ttl=86400)
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
    if value <= 12.0: return "<span class='good-badge'>🟢 Good (Healthy)</span>"
    elif value <= 35.4: return "<span class='mod-badge'>🟡 Moderate (Caution)</span>"
    else: return "<span class='unh-badge'>🔴 Unhealthy (At Risk)</span>"

# ============================================
# DATA PIPELINE & ML MODEL
# ============================================
# Load historical data
try:
    df_yearly = pd.read_csv("wake_pm25_by_year.csv").sort_values("year")
except:
    df_yearly = pd.DataFrame(columns=["year", "mean_pm25"])

current_year = datetime.now().year

# Fetch current year
with st.spinner("Syncing with EPA AirData..."):
    df_daily = fetch_pm25(current_year)

if df_daily is not None and not df_yearly.empty:
    mean_pm25 = df_daily["arithmetic_mean"].mean()
    new_row = pd.DataFrame({"year": [current_year], "mean_pm25": [mean_pm25]})
    df_yearly = df_yearly[df_yearly["year"] != current_year]
    df_yearly = pd.concat([df_yearly, new_row], ignore_index=True).sort_values("year")

# Train Model
if not df_yearly.empty:
    X = df_yearly[["year"]]
    y = df_yearly["mean_pm25"]
    model = LinearRegression().fit(X, y)
    
    future_years = np.arange(current_year + 1, current_year + 11)
    future_preds = model.predict(future_years.reshape(-1, 1))
    future_df = pd.DataFrame({"year": future_years, "predicted_pm25": future_preds})

# ============================================
# APP LAYOUT & NAVIGATION
# ============================================
# Hero Section
st.markdown('<p class="big-font">Wake County Air Quality Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Predictive analytics and health literacy for atmospheric particulate matter (PM2.5).</p>', unsafe_allow_html=True)

# App Navigation using Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview & Forecast", "🗃️ Data Explorer", "🩺 Health Literacy & FAQ"])

# --------------------------------------------
# TAB 1: OVERVIEW & FORECAST
# --------------------------------------------
with tab1:
    if not df_yearly.empty:
        current_val = y.iloc[-1]
        
        # Metric Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>Current {current_year} Average</h3><h2>{current_val:.2f} µg/m³</h2>{get_aqi_badge(current_val)}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Model Accuracy (R²)</h3><h2>{model.score(X,y):.3f}</h2><span>1.0 is a perfect fit</span></div>", unsafe_allow_html=True)
        with col3:
            trend = "Improving" if model.coef_[0] < 0 else "Worsening"
            st.markdown(f"<div class='metric-card'><h3>10-Year Trend</h3><h2>{trend}</h2><span>Slope: {model.coef_[0]:.4f}</span></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Advanced Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_yearly["year"], y=df_yearly["mean_pm25"], mode="markers+lines", name="Historical Avg", line=dict(color="#1E3A8A", width=3), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=df_yearly["year"], y=model.predict(X), mode="lines", name="Trendline", line=dict(color="gray", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=future_df["year"], y=future_df["predicted_pm25"], mode="markers+lines", name="10-Year Forecast", line=dict(color="#EF4444", width=3, dash="dot"), marker=dict(size=10, symbol="diamond")))
        
        fig.update_layout(title="PM2.5 Concentration: Historical vs. Predicted", xaxis_title="Year", yaxis_title="Mean PM2.5 (µg/m³)", hovermode="x unified", template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Awaiting historical data to generate models.")

# --------------------------------------------
# TAB 2: DATA EXPLORER
# --------------------------------------------
with tab2:
    st.subheader("Raw Data & Export")
    st.write("Examine the underlying datasets driving the machine learning model.")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Historical Records**")
        st.dataframe(df_yearly, use_container_width=True, hide_index=True)
        # CSV Download Button
        csv_hist = df_yearly.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Historical CSV", data=csv_hist, file_name="wake_historical_pm25.csv", mime="text/csv")
        
    with colB:
        st.markdown("**Forecasting Data**")
        st.dataframe(future_df, use_container_width=True, hide_index=True)
        csv_fut = future_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", data=csv_fut, file_name="wake_forecast_pm25.csv", mime="text/csv")

# --------------------------------------------
# TAB 3: HEALTH LITERACY & FAQ
# --------------------------------------------
with tab3:
    st.subheader("Glossary & Medical Context")
    
    with st.expander("🔬 What exactly is PM2.5?"):
        st.write("PM2.5 refers to fine inhalable particles with diameters that are generally 2.5 micrometers and smaller. To put that in perspective, a single human hair is about 30 times larger than the largest fine particle.")
        
    with st.expander("🫁 How does it affect human health?"):
        st.write("""
        Because they are so small, PM2.5 particles can travel deeply into the respiratory tract, reaching the lungs and potentially entering the bloodstream. 
        * **Short-term exposure:** Can cause irritation of the eyes, nose, and throat, coughing, sneezing, and shortness of breath.
        * **Long-term exposure:** Associated with increased rates of chronic bronchitis, reduced lung function, and increased mortality from lung cancer and heart disease.
        """)
        
    with st.expander("📉 How to reduce exposure"):
        st.write("""
        1. **Check the AQI:** Monitor this dashboard or AirNow.gov before outdoor exercise.
        2. **Stay Indoors:** On 'Unhealthy' days, keep windows closed and run air purifiers with HEPA filters.
        3. **Mask Up:** If you must be outside during severe events (like wildfire smoke), wear a well-fitted N95 or KN95 mask.
        """)

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>Data sourced directly from the United States Environmental Protection Agency (EPA) AirData API. Developed for environmental health research.</p>", unsafe_allow_html=True)
