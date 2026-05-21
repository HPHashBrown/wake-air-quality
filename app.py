import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ============================================
# CONFIG — ENTER YOUR EPA API INFO
# ============================================
EMAIL = "harshuaz11@gmail.com"
API_KEY = "goldswift82"

st.title("Wake County PM2.5 Forecasting Dashboard")
st.write("This dashboard automatically updates using EPA AirData.")

# ============================================
# FUNCTION TO FETCH DAILY PM2.5
# ============================================
def fetch_pm25(year):
    url = (
        f"https://aqs.epa.gov/data/api/dailyData/byCounty?"
        f"email={EMAIL}&key={API_KEY}"
        f"&param=88101&bdate={year}0101&edate={year}1231"
        f"&state=37&county=183"
    )
    response = requests.get(url).json()

    if "Data" not in response:
        return None

    df = pd.DataFrame(response["Data"])
    if df.empty:
        return None

    return df[["date_local", "arithmetic_mean"]]

# ============================================
# LOAD EXISTING DATA
# ============================================
df_yearly = pd.read_csv("wake_pm25_by_year.csv")
df_yearly = df_yearly.sort_values("year")

st.subheader("Historical Yearly PM2.5 Data")
st.dataframe(df_yearly)

# ============================================
# UPDATE CURRENT YEAR
# ============================================
current_year = datetime.now().year
df_daily = fetch_pm25(current_year)

if df_daily is not None:
    mean_pm25 = df_daily["arithmetic_mean"].mean()

    new_row = pd.DataFrame({
        "year": [current_year],
        "mean_pm25": [mean_pm25]
    })

    df_yearly = df_yearly[df_yearly["year"] != current_year]
    df_yearly = pd.concat([df_yearly, new_row], ignore_index=True)
    df_yearly = df_yearly.sort_values("year")

st.subheader("Updated Dataset")
st.dataframe(df_yearly)

# ============================================
# REGRESSION MODEL
# ============================================
X = df_yearly[["year"]]
y = df_yearly["mean_pm25"]

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(X, y)

st.subheader("Model Statistics")
st.write(f"**Slope:** {slope:.4f}")
st.write(f"**Intercept:** {intercept:.4f}")
st.write(f"**R²:** {r2:.3f}")

# ============================================
# FUTURE PREDICTIONS
# ============================================
future_years = np.arange(current_year + 1, current_year + 11)
future_preds = model.predict(future_years.reshape(-1, 1))

future_df = pd.DataFrame({
    "year": future_years,
    "predicted_pm25": future_preds
})

st.subheader("Future Predictions (Next 10 Years)")
st.dataframe(future_df)

# ============================================
# PLOT
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df_yearly["year"], df_yearly["mean_pm25"], color="blue", label="Historical")
ax.plot(df_yearly["year"], model.predict(X), color="black", label="Trendline")
ax.scatter(future_years, future_preds, color="red", label="Predicted")

ax.set_xlabel("Year")
ax.set_ylabel("Mean PM2.5 (µg/m³)")
ax.set_title("Wake County PM2.5 Forecasting Model")
ax.grid(True)
ax.legend()

st.pyplot(fig)
