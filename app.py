import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 1. Page Settings ----------------
st.set_page_config(page_title="Sepsis Prediction Model", layout="wide")
st.title("Predictive Modeling of Sepsis Risk in Diabetic Stroke Patients")
st.markdown("Risk Assessment System based on Machine Learning Algorithm")

# ---------------- 2. Load the Model ----------------
# ⚠️ 已经修改为你当前的文件名：RandomForest.pkl
@st.cache_resource
def load_model():
    try:
        with open("RandomForest.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# ---------------- 3. Sidebar Input Panel ----------------
st.sidebar.header("📝 Input Patient Parameters")

def user_input_features():
    # Categorical variables (Dropdown menu)
    aki = st.sidebar.selectbox("AKI (Acute Kidney Injury)", options=[0, 1], format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)")
    pneumonia = st.sidebar.selectbox("Pneumonia", options=[0, 1], format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)")
    mv = st.sidebar.selectbox("MV (Mechanical Ventilation)", options=[0, 1], format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)")
    
    # Continuous variables (Sliders)
    gcs = st.sidebar.slider("GCS (Glasgow Coma Scale)", min_value=3.0, max_value=15.0, value=15.0)
    platelet_count = st.sidebar.slider("Platelet count", min_value=0.0, max_value=1000.0, value=200.0)
    rbc = st.sidebar.slider("RBC", min_value=1.0, max_value=10.0, value=4.5)
    wbc = st.sidebar.slider("WBC", min_value=0.0, max_value=50.0, value=10.0)
    anion_gap = st.sidebar.slider("Anion_gap", min_value=0.0, max_value=40.0, value=12.0)
    chloride = st.sidebar.slider("Chloride", min_value=70.0, max_value=130.0, value=100.0)
    pt = st.sidebar.slider("PT (Prothrombin Time)", min_value=8.0, max_value=30.0, value=12.0)
    scr = st.sidebar.slider("scr (Serum Creatinine)", min_value=0.1, max_value=15.0, value=1.0)
    diastolic_dbp = st.sidebar.slider("Diastolic DBP", min_value=30.0, max_value=150.0, value=80.0)
    mean_bp = st.sidebar.slider("Mean BP", min_value=40.0, max_value=160.0, value=90.0)
    respiratory = st.sidebar.slider("Respiratory Rate", min_value=5.0, max_value=50.0, value=18.0)
    temperature = st.sidebar.slider("Temperature", min_value=30.0, max_value=42.0, value=37.0)

    # Dictionary keys matching training data exactly
    data = {
        'AKI': aki,
        'Pneumonia': pneumonia,
        'MV': mv,
        'GCS': gcs,
        'Platelet count': platelet_count,
        'RBC': rbc,
        'WBC': wbc,
        'Anion_gap': anion_gap,
        'Chloride': chloride,
        'PT': pt,
        'scr': scr,
        'Diastolic DBP': diastolic_dbp,
        'Mean BP': mean_bp,
        'Respiratory': respiratory,
        'Temperature': temperature
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ---------------- 4. Main Panel ----------------
st.write("### Current Patient Parameters Overview")
st.dataframe(input_df, hide_index=True)

if model is None:
    st.error("⚠️ Model file 'RandomForest.pkl' not found. Please ensure the model file is in the same directory as this script.")
else:
    # Predict button
    if st.button("🚀 Predict", type="primary"):
        
        # 🌟 Automatically align the columns to the model's expected order
        if hasattr(model, "feature_names_in_"):
            input_df = input_df[model.feature_names_in_]
            
        # Get prediction probability
        pred_prob = model.predict_proba(input_df)[0][1]
        
        # Set the threshold
        threshold = 0.50
        risk_label = "High Risk of Sepsis" if pred_prob >= threshold else "Low Risk of Sepsis"
        
        st.write("---")
        st.write(f"### Prediction Result: **{risk_label}**")
        
        # ---------------- 5. Draw Donut Chart ----------------
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ratios = [pred_prob, 1 - pred_prob]
        labels = ['Sepsis Risk', 'Non-Sepsis']
        colors = ['#ED0000', '#0099B4'] 
        
        wedges, texts, autotexts = ax.pie(
            ratios, 
            colors=colors, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90, 
            pctdistance=0.85,
            textprops={'fontsize': 14, 'weight': 'bold'}
        )
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        
        ax.text(0, 0, f"Risk Prob\n{pred_prob*100:.1f}%", ha='center', va='center', fontsize=18, fontweight='bold', color='black')
        
        ax.axis('equal') 
        st.pyplot(fig)
