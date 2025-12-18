# app.py
import streamlit as st
from prediction import predict_damage

st.set_page_config(page_title="Earthquake Damage Predictor", layout="wide")

st.title("Earthquake Building Damage Prediction")
st.write("Input building features to predict potential earthquake damage.")

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")

with st.sidebar.form("input_form"):
    age_building = st.number_input("Building Age (years)", 0, 200, 20)
    count_floors_pre_eq = st.number_input("Number of Floors", 1, 10, 2)
    
    foundation_type = st.selectbox("Foundation Type", ['Cement-Stone/Brick', 'Mud mortar-Stone/Brick', 'RC', 'Other'])
    roof_type = st.selectbox("Roof Type", ['Bamboo/Timber-Light roof', 'RCC/RB/RBC'])
    ground_floor_type = st.selectbox("Ground Floor Type", ['Mud', 'RC', 'Other', 'Timber'])
    
    position = st.selectbox("Position", ['Attached-2 side', 'Attached-3 side', 'Not attached'])
    land_surface_condition = st.selectbox("Land Surface Condition", ['Flat', 'Moderate slope', 'Steep slope'])
    
    submit = st.form_submit_button("Predict")

# --- Main Panel ---
if submit:
    user_input = {
        'age_building': age_building,
        'count_floors_pre_eq': count_floors_pre_eq,
        'foundation_type': foundation_type,
        'roof_type': roof_type,
        'ground_floor_type': ground_floor_type,
        'position': position,
        'land_surface_condition': land_surface_condition
    }

    label, confidence = predict_damage(user_input)

    st.success(f"Predicted Damage Level: **{label}**")
    st.write(f"Model Confidence: **{confidence:.2%}**")
