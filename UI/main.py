import joblib
import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Heart Disease Survival Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------- Load Model ----------------
model = joblib.load("../Model/Heart_disease_model.pkl")

# ---------------- Title ----------------
st.title("❤️ Heart Disease Survival Predictor")
st.markdown("Predict whether a heart failure patient is **likely to survive** based on clinical records.")

st.divider()

# ---------------- Input Form ----------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 95, 50)

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        anaemia = st.checkbox("Anaemia")
        diabetes = st.checkbox("Diabetes")
        smoking = st.checkbox("Smoking")

    with col2:
        high_blood_pressure = st.checkbox("High Blood Pressure")

        creatinine_phosphokinase = st.slider(
            "Creatinine Phosphokinase",
            23,
            7861,
            250
        )

        ejection_fraction = st.slider(
            "Ejection Fraction (%)",
            14,
            80,
            38
        )

    st.divider()

    col3, col4, col5 = st.columns(3)

    with col3:
        platelets = st.slider(
            "Platelets",
            25100,
            850000,
            263358
        )

    with col4:
        serum_creatinine = st.slider(
            "Serum Creatinine",
            0.5,
            9.4,
            1.1
        )

    with col5:
        serum_sodium = st.slider(
            "Serum Sodium",
            113,
            148,
            137
        )

    submit = st.form_submit_button("🔍 Predict Survival")

# ---------------- Prediction ----------------

if submit:

    input_data = [[
        age,
        int(anaemia),
        creatinine_phosphokinase,
        int(diabetes),
        ejection_fraction,
        int(high_blood_pressure),
        platelets,
        serum_creatinine,
        serum_sodium,
        0 if sex == "Male" else 1,
        int(smoking)
    ]]

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("🟢 The patient is likely to survive.")
        st.progress(float(probability[0]))
        st.metric("Survival Probability", f"{probability[0]*100:.2f}%")

    else:
        st.error("🔴 The patient has a high risk of death.")
        st.progress(float(probability[1]))
        st.metric("Risk Probability", f"{probability[1]*100:.2f}%")

    with st.expander("📋 Input Summary"):
        st.write({
            "Age": age,
            "Sex": sex,
            "Anaemia": anaemia,
            "Diabetes": diabetes,
            "Smoking": smoking,
            "High Blood Pressure": high_blood_pressure,
            "Creatinine Phosphokinase": creatinine_phosphokinase,
            "Ejection Fraction": ejection_fraction,
            "Platelets": platelets,
            "Serum Creatinine": serum_creatinine,
            "Serum Sodium": serum_sodium
        })