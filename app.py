import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---- LOAD & PREPARE DATA ----
@st.cache_data
def load_and_train():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    df = pd.read_csv(url)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour']  = df['date_time'].dt.hour
    df['day']   = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df = df[df['temp'] > 200]
    df = df[df['rain_1h'] < 500]
    df['holiday'] = df['holiday'].fillna('None')

    le = LabelEncoder()
    df['weather_encoded']  = le.fit_transform(df['weather_main'])
    df['is_holiday']  = (df['holiday'] != 'None').astype(int)
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 16 <= x <= 19) else 0)
    df['is_weekend']  = (df['day'] >= 5).astype(int)

    features = ['hour','day','month','temp','rain_1h',
                'clouds_all','weather_encoded','is_holiday',
                'is_rush_hour','is_weekend']
    X = df[features]
    y = df['traffic_volume']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, le

# ---- PAGE SETUP ----
st.title("🚦 Smart City Traffic Predictor")
st.markdown("Predict traffic volume based on time and weather conditions")

with st.spinner("Training model on real traffic data... (~30 sec)"):
    model, le = load_and_train()

st.success("Model ready!")

# ---- SIDEBAR INPUTS ----
st.sidebar.header("Set Conditions")

hour    = st.sidebar.slider("Hour of Day", 0, 23, 8)
day     = st.sidebar.selectbox("Day of Week", 
            [0,1,2,3,4,5,6], 
            format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month   = st.sidebar.slider("Month", 1, 12, 1)
temp    = st.sidebar.slider("Temperature (Kelvin)", 230, 310, 280)
rain    = st.sidebar.slider("Rain (mm/hr)", 0.0, 50.0, 0.0)
clouds  = st.sidebar.slider("Cloud Cover (%)", 0, 100, 20)
weather = st.sidebar.selectbox("Weather", ['Clear','Clouds','Rain','Snow','Mist','Drizzle','Fog','Thunderstorm'])
holiday = st.sidebar.checkbox("Is it a Holiday?")

# ---- PREDICT ----
is_rush  = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
is_wkend = 1 if day >= 5 else 0

try:
    w_encoded = le.transform([weather])[0]
except:
    w_encoded = 0

sample = pd.DataFrame([{
    'hour': hour, 'day': day, 'month': month,
    'temp': temp, 'rain_1h': rain, 'clouds_all': clouds,
    'weather_encoded': w_encoded, 'is_holiday': int(holiday),
    'is_rush_hour': is_rush, 'is_weekend': is_wkend
}])

prediction = model.predict(sample)[0]

# ---- DISPLAY ----
st.subheader("Predicted Traffic Volume")

if prediction > 4500:
    st.error(f"🔴 HIGH CONGESTION — {prediction:.0f} vehicles/hour")
elif prediction > 2500:
    st.warning(f"🟡 MODERATE TRAFFIC — {prediction:.0f} vehicles/hour")
else:
    st.success(f"🟢 LIGHT TRAFFIC — {prediction:.0f} vehicles/hour")

st.metric("Vehicles per Hour", f"{prediction:.0f}")

col1, col2, col3 = st.columns(3)
col1.metric("Rush Hour", "Yes" if is_rush else "No")
col2.metric("Weekend", "Yes" if is_wkend else "No")
col3.metric("Holiday", "Yes" if holiday else "No")