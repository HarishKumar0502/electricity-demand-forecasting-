import streamlit as st
import pandas as pd
import joblib
import requests
from dotenv import load_dotenv
import os
api=load_dotenv(".env")

# -------------------------
# Load model
# -------------------------
model = joblib.load("electicity_xgb_prediction_model.pkl")

# -------------------------
# Load historical data (for patterns)
# -------------------------
data = pd.read_csv("data.csv")
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

# create time features
data['hour'] = data['Timestamp'].dt.hour
data['dayofweek'] = data['Timestamp'].dt.dayofweek
data['month'] = data['Timestamp'].dt.month

# -------------------------
# Weather API
# -------------------------
API_KEY = os.getenv("WAPI")  
# 🔑 replace this

def get_weather(city, future_time):
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()

        if response["cod"] != "200":
            return None, None

        closest = response["list"][0]
        min_diff = float("inf")

        # find closest forecast time
        for item in response["list"]:
            forecast_time = pd.to_datetime(item["dt_txt"])
            diff = abs((forecast_time - future_time).total_seconds())

            if diff < min_diff:
                min_diff = diff
                closest = item

        temp = closest["main"]["temp"]
        humidity = closest["main"]["humidity"]

        return temp, humidity

    except:
        return None, None

# -------------------------
# Streamlit UI
# -------------------------
st.title("⚡ Electricity Demand Forecast (Real-Time)")

st.write("ML-based electricity demand prediction with live weather")

date = st.date_input("📅 Select Date")
hour = st.slider("⏰ Hour", 0, 23, 12)

city = st.text_input("🌍 Enter City", "Erode")

# -------------------------
# Create timestamp
# -------------------------
future_time = pd.to_datetime(date) + pd.Timedelta(hours=hour)

# -------------------------
# Time features
# -------------------------
year = future_time.year
month = future_time.month
dayofweek = future_time.dayofweek
dayofyear = future_time.dayofyear
weekofyear = future_time.isocalendar().week
quarter = (month - 1)//3 + 1
is_weekend = 1 if dayofweek >= 5 else 0

# -------------------------
# Get weather
# -------------------------
temperature, humidity = get_weather(city, future_time)

if temperature is None:
    st.error("❌ Weather fetch failed. Check city name or API key.")
    st.stop()

st.info(f"🌡 Temperature in {city}: {temperature} °C")
st.info(f"💧 Humidity in {city}: {humidity} %")

# -------------------------
# Pattern-based lag
# -------------------------
lag24 = data[
    (data['hour'] == hour) &
    (data['month'] == month)
]['Demand'].mean()

lagweek = data[
    (data['dayofweek'] == dayofweek)
]['Demand'].mean()

rolling_mean = data['Demand'].mean()
rolling_std = data['Demand'].std()

# -------------------------
# Input DataFrame
# -------------------------
input_data = pd.DataFrame({
    'hour':[hour],
    'dayofweek':[dayofweek],
    'month':[month],
    'year':[year],
    'dayofyear':[dayofyear],
    'weekofyear':[weekofyear],
    'quarter':[quarter],
    'is_weekend':[is_weekend],
    'Temperature':[temperature],
    'Humidity':[humidity],
    'Demand_lag_24hr':[lag24],
    'Demand_lag_week':[lagweek],
    'demand_Rolling_mean_24hr':[rolling_mean],
    'demand_Rolling_std_24hr':[rolling_std]
})

# -------------------------
# Prediction
# -------------------------
if st.button("🚀 Predict Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"⚡ Predicted Electricity Demand: {prediction:.2f}")