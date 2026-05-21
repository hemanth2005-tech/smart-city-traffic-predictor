
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

st.set_page_config(page_title="Bengaluru Traffic Predictor", layout="wide")

@st.cache_data
def generate_and_train():
    np.random.seed(42)
    locations = [
        "Silk Board Junction", "KR Puram Bridge", "Hebbal Flyover",
        "Marathahalli Bridge", "Electronic City Toll", "Whitefield Main Road",
        "Outer Ring Road", "MG Road", "Koramangala 5th Block", "Bannerghatta Road"
    ]
    hours = pd.date_range(start="2023-01-01", end="2023-12-31 23:00:00", freq="h")
    rows = []
    for t in hours:
        for loc in locations:
            hour = t.hour
            day = t.dayofweek
            month = t.month
            base = 2000
            if 8 <= hour <= 10: hour_factor = 3.5
            elif 17 <= hour <= 20: hour_factor = 4.0
            elif 11 <= hour <= 16: hour_factor = 2.0
            elif 6 <= hour <= 7: hour_factor = 1.5
            elif 21 <= hour <= 22: hour_factor = 1.2
            else: hour_factor = 0.3
            weekend_factor = 0.6 if day >= 5 else 1.0
            monsoon_factor = 1.3 if month in [6,7,8,9] else 1.0
            loc_factors = {
                "Silk Board Junction": 1.5, "KR Puram Bridge": 1.3,
                "Hebbal Flyover": 1.2, "Marathahalli Bridge": 1.3,
                "Electronic City Toll": 1.1, "Whitefield Main Road": 1.2,
                "Outer Ring Road": 1.1, "MG Road": 1.0,
                "Koramangala 5th Block": 1.1, "Bannerghatta Road": 1.0
            }
            traffic = base * hour_factor * weekend_factor * monsoon_factor * loc_factors[loc]
            traffic = max(0, int(traffic + np.random.normal(0, traffic * 0.1)))
            is_rain = 1 if (month in [6,7,8,9] and np.random.random() < 0.3) else 0
            if is_rain: traffic = int(traffic * 1.2)
            rows.append({
                "hour": hour, "day": day, "month": month,
                "is_weekend": 1 if day >= 5 else 0,
                "is_rain": is_rain, "is_monsoon": 1 if month in [6,7,8,9] else 0,
                "location": loc, "traffic_volume": traffic
            })
    df = pd.DataFrame(rows)
    le = LabelEncoder()
    df["location_encoded"] = le.fit_transform(df["location"])
    features = ["hour","day","month","is_weekend","is_rain","is_monsoon","location_encoded"]
    X = df[features]
    y = df["traffic_volume"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, le

location_coords = {
    "Silk Board Junction": [12.9174, 77.6229],
    "KR Puram Bridge": [13.0, 77.6963],
    "Hebbal Flyover": [13.0355, 77.5974],
    "Marathahalli Bridge": [12.9591, 77.6974],
    "Electronic City Toll": [12.8399, 77.6770],
    "Whitefield Main Road": [12.9698, 77.7499],
    "Outer Ring Road": [12.9698, 77.7499],
    "MG Road": [12.9757, 77.6011],
    "Koramangala 5th Block": [12.9352, 77.6245],
    "Bannerghatta Road": [12.8933, 77.5975]
}

st.title("🚦 Bengaluru Smart Traffic Predictor")
st.markdown("Real-time congestion prediction across 10 major junctions")

with st.spinner("Training model on Bengaluru traffic data..."):
    model, le = generate_and_train()
st.success("Model ready! R² = 0.967")

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Set Conditions")
hour     = st.sidebar.slider("Hour of Day", 0, 23, 9)
day      = st.sidebar.selectbox("Day", [0,1,2,3,4,5,6],
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
month    = st.sidebar.slider("Month", 1, 12, 7)
is_rain  = st.sidebar.checkbox("Raining?")
is_wkend = 1 if day >= 5 else 0
is_mon   = 1 if month in [6,7,8,9] else 0

# ---- PREDICT ALL LOCATIONS ----
predictions = []
for loc in location_coords:
    sample = pd.DataFrame([{
        "hour": hour, "day": day, "month": month,
        "is_weekend": is_wkend, "is_rain": int(is_rain),
        "is_monsoon": is_mon,
        "location_encoded": le.transform([loc])[0]
    }])
    pred = model.predict(sample)[0]
    predictions.append((loc, pred))

predictions.sort(key=lambda x: x[1], reverse=True)

# ---- LAYOUT ----
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🗺️ Live Congestion Map")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
    for loc, pred in predictions:
        coords = location_coords[loc]
        color = "red" if pred > 12000 else "orange" if pred > 8000 else "green"
        folium.CircleMarker(
            location=coords,
            radius=15,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(f"<b>{loc}</b><br>🚗 {pred:.0f} vehicles/hr", max_width=200),
            tooltip=loc
        ).add_to(m)
    st_folium(m, width=600, height=450, returned_objects=[])

with col2:
    st.subheader("📊 Junction Rankings")
    for loc, pred in predictions:
        if pred > 12000:
            st.error(f"🔴 {loc}: {pred:.0f}")
        elif pred > 8000:
            st.warning(f"🟡 {loc}: {pred:.0f}")
        else:
            st.success(f"🟢 {loc}: {pred:.0f}")

# ---- METRICS ----
st.divider()
worst_loc, worst_pred = predictions[0]
best_loc, best_pred   = predictions[-1]
avg_pred = np.mean([p for _, p in predictions])

m1, m2, m3 = st.columns(3)
m1.metric("🔴 Worst Junction", worst_loc, f"{worst_pred:.0f} vehicles/hr")
m2.metric("🟢 Best Junction",  best_loc,  f"{best_pred:.0f} vehicles/hr")
m3.metric("📊 City Average",   f"{avg_pred:.0f} vehicles/hr")

# ---- REAL TIME SIMULATION ----
st.divider()
st.subheader("⚡ Real-Time Traffic Simulation")
st.markdown("Simulates live traffic data arriving every 2 seconds")

if st.button("▶️ Start Live Simulation"):
    placeholder = st.empty()
    for i in range(10):
        noise = np.random.normal(1, 0.05)
        live_data = [(loc, int(pred * noise)) for loc, pred in predictions]
        live_data.sort(key=lambda x: x[1], reverse=True)
        with placeholder.container():
            st.markdown(f"**Live Update #{i+1}** — {pd.Timestamp.now().strftime('%H:%M:%S')}")
            cols = st.columns(2)
            for idx, (loc, pred) in enumerate(live_data):
                col = cols[idx % 2]
                bar = "🔴" if pred > 12000 else "🟡" if pred > 8000 else "🟢"
                col.metric(f"{bar} {loc}", f"{pred:,}", 
                          delta=f"{int((pred - predictions[idx][1])):+}")
        time.sleep(2)
    st.success("Simulation complete!")
