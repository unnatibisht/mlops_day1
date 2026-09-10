import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/champion_model.pkl")

st.title("Advertising Sales Prediction")

tv = st.number_input("TV Budget")
radio = st.number_input("Radio Budget")
newspaper = st.number_input("Newspaper Budget")

if st.button("Prediction Sales"):

    input_data = pd.DataFrame({
        "TV": [tv],
        "radio": [radio],
        "newspaper": [newspaper]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted sales: {prediction[0]:.2f}"
    )
