import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Wake County Air Quality",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SIDEBAR: SECURE CONFIG & INFO
# ============================================
with st.sidebar:
    st.title("⚙️ Dashboard Settings")
    st.markdown("Manage your EPA AirData API credentials below.")
    
    # Moving credentials to inputs prevents hardcoding sensitive info
    email = st.text_input("EPA Email", value="harshuaz11@gmail.com")
    api_key = st.text_input("EPA API Key", value="goldswift82", type="password")
    
    st.divider()
    st.markdown("""
    **About this Dashboard:**
    Automatically fetches daily PM2.5 levels for Wake County, NC, and uses a Linear Regression model to forecast trends over the next decade.
    """)

# ============================================
# MAIN HEADER
# ============================================
st.title("🌤️ Wake County PM2.5 Forecasting Dashboard")
st.markdown("Track historical air quality trends and explore predictive machine learning models for future particulate matter levels.")
st.divider()

# ============================================
# FUNCTIONS (WITH CACHING FOR SPEED)
# ============================================
@st.cache_data(ttl=86400) # Caches the data for 24 hours so it doesn't slow down your app
def fetch_pm25(year, user_email, user_key):
    url = (
        f"https://aqs.epa.gov/data/api/dailyData/byCounty?"
        f"email={user_email}&key={user_key}"
        f"&param=88101&bdate={year}0101&edate={year}1231"
        f"&state=37&county=183"
    )
    try:
        response = requests.get(url).json()
        if "Data" not in response:
            return None
        df = pd.DataFrame(response["Data"])
        if df.empty:
            return None
        return df[["date_local", "arithmetic_mean"]]
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("wake_pm25_by_year.csv")
        return df.sort_values("year")
    except FileNotFoundError:
        st.warning("⚠️ 'wake_pm25_by_year.csv' not found. Please ensure it's in the same directory.")
        return pd.DataFrame(columns=["year", "mean_pm25"])

# ============================================
# DATA PROCESSING
# ============================================
df_yearly = load_historical_data()
current_year = datetime.now().year

with st.spinner("Fetching latest EPA data..."):
    df_daily = fetch_pm25(current_year, email, api_key)

if df_daily is not None and not df_yearly.empty:
    mean_pm25 = df_daily["arithmetic_mean"].mean()
    new_row = pd.DataFrame({"year": [current_year], "mean_pm25": [mean_pm25]})
    
    df_yearly = df_yearly[df_yearly["year"] != current_year]
    df_yearly = pd.concat([df_yearly, new_row], ignore_index=True)
    df_yearly = df_yearly.sort_values("year")

# ============================================
# MACHINE LEARNING MODEL
# ============================================
if not df_yearly.empty:
    X = df_yearly[["year"]]
    y = df_yearly["mean_pm25"]

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    # Predictions
    future_years = np.arange(current_year + 1, current_year + 11)
    future_preds = model.predict(future_years.reshape(-1, 1))
    future_df = pd.DataFrame({"year": future_years, "predicted_pm25": future_preds})

    # ============================================
    # UI: TOP METRICS ROW
    # ============================================
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"Current {current_year} PM2.5", value=f"{y.iloc[-1]:.2f} µg/m³", 
                  delta=f"{y.iloc[-1] - y.iloc[-2]:.2f} from last year" if len(y) > 1 else None,
                  delta_color="inverse") # Inverse because lower PM2.5 is better
    with col2:
        st.metric(label="Model R² Score", value=f"{r2:.3f}", 
                  help="Closer to 1.0 means a stronger fit to the historical data.")
    with col3:
        st.metric(label="Annual Trend (Slope)", value=f"{slope:.4f}",
                  help="Negative value indicates air quality is improving over time.")

    st.divider()

    # ============================================
    # UI: INTERACTIVE PLOTLY CHART
    # ============================================
    st.subheader("📊 PM2.5 Trends & 10-Year Forecast")
    
    fig = go.Figure()

    # 1. Historical Scatter & Lines
    fig.add_trace(go.Scatter(
        x=df_yearly["year"], y=df_yearly["mean_pm25"],
        mode="markers+lines", name="Historical Average",
        marker=dict(color="#1f77b4", size=8),
        line=dict(color="#1f77b4", width=2)
    ))

    # 2. Regression Trendline
    fig.add_trace(go.Scatter(
        x=df_yearly["year"], y=model.predict(X),
        mode="lines", name="Linear Trend",
        line=dict(color="rgba(0,0,0,0.4)", width=2, dash="dash")
    ))

    # 3. Future Predictions
    fig.add_trace(go.Scatter(
        x=future_df["year"], y=future_df["predicted_pm25"],
        mode="markers+lines", name="Predicted Forecast",
        marker=dict(color="#d62728", size=8, symbol="diamond"),
        line=dict(color="#d62728", width=2, dash="dot")
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Mean PM2.5 (µg/m³)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # UI: DATA TABLES (Side-by-side expanders)
    # ============================================
    st.markdown("### 🗃️ Raw Data Explorer")
    t_col1, t_col2 = st.columns(2)
    
    with t_col1:
        with st.expander("View Historical Dataset"):
            st.dataframe(df_yearly, use_container_width=True, hide_index=True)
            
    with t_col2:
        with st.expander("View 10-Year Predictions"):
            st.dataframe(future_df, use_container_width=True, hide_index=True)

else:
    st.info("Awaiting historical data to train the model and generate charts.")
