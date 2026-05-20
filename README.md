# 🚦 Smart City Traffic Predictor

A machine learning web app that predicts urban traffic congestion based on time, weather, and road conditions.

## 🎯 What it does
- Predicts traffic volume (vehicles/hour) for any time and weather condition
- Classifies congestion as 🔴 High / 🟡 Moderate / 🟢 Light
- Interactive dashboard with real-time predictions

## 🛠️ Tech Stack
- **Python** — core language
- **Pandas & NumPy** — data cleaning and engineering
- **Scikit-learn** — Random Forest ML model (95.2% accuracy)
- **Matplotlib** — data visualization
- **Streamlit** — interactive web dashboard

## 📊 Dataset
Real traffic data from Minneapolis highway (48,000+ hourly records, 2012-2018)  
Source: UCI Machine Learning Repository

## 🚀 How to Run
```bash
pip install pandas numpy matplotlib scikit-learn streamlit
streamlit run app.py
```

## 📈 Model Performance
| Metric | Score |
|--------|-------|
| R² Score | 0.952 |
| MAE | 243 vehicles/hour |

## 💡 Key Insights from Data
- Rush hours (8am & 5pm) have 4x more traffic than off-peak hours
- Weekday traffic is significantly higher than weekends
- Rain and snow reduce traffic volume noticeably

## 🔮 Future Improvements
- Integrate Indian city traffic data (Bengaluru, Hyderabad)
- Add live map visualization with folium
- Real-time data streaming with Apache Kafka
