import joblib
import streamlit as st
import os

model_path = os.path.join("Heart_disease_model.pkl")
model = joblib.load(model_path)

st.title("Titanic Survival Predictor")