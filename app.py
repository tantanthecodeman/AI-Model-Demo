import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


rf_model = joblib.load(".src/pump_rf_model.pkl")  

st.title("Solar-Powered Dewatering Pump")
st.markdown("Simulate pump ON/OFF based on water level, rainfall, solar, and time of day.")

water = st.slider("Water Level (m)", 0.0, 10.0, 5.0)
rain = st.slider("Rainfall Forecast (mm)", 0, 100, 30)
solar = st.slider("Solar Irradiance (W/m²)", 0, 1000, 400)
time = st.selectbox("Time of Day", [0,1,2,3], format_func=lambda x: ["Night","Morning","Noon","Evening"][x])
diesel = st.slider("Diesel Cost (₹/kWh)", 10, 20, 15)

input_features = pd.DataFrame([[water, rain, solar, time, diesel]],
                              columns=["water_level","rain_forecast","solar_irradiance","time_of_day","diesel_cost"])

prediction = rf_model.predict(input_features)[0]

st.markdown(f"### Pump Status: {'🟢 ON' if prediction==1 else '🔴 OFF'}")

st.subheader("Feature Importance")
importances = rf_model.feature_importances_
features = input_features.columns

fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
st.pyplot(fig)

st.subheader("Input & Predicted Output")
st.table(pd.concat([input_features, pd.DataFrame({'pump_prediction':[prediction]})], axis=1))



