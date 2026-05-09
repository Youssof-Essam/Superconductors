import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Superconductivity Predictor", layout="wide")

st.title("🧪 Superconductivity Critical Temperature Predictor")
st.markdown("""
This dashboard uses a **PCA-optimized SVR model** to predict the critical temperature (K) 
of materials based on their chemical and physical properties.
""")

# API Configuration (Use the Docker service name)
API_URL = "http://api:8000/predict"

# Sidebar for Input Method
st.sidebar.header("Input Method")
input_mode = st.sidebar.radio("Choose Input:", ["Manual Features", "Batch Upload (CSV)"])

if input_mode == "Manual Features":
    st.subheader("Manual Feature Input")
    st.info("Note: The model expects 81 features. For demonstration, we'll use a random vector.")
    
    # In a real scenario, you'd create sliders or number inputs for key features.
    # Here we simulate the 81-feature vector.
    if st.button("Generate Random Material & Predict"):
        random_features = np.random.rand(81).tolist()
        
        with st.spinner("Calling Prediction API..."):
            response = requests.post(API_URL, json={"data": random_features})
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"### Predicted Critical Temperature: {result['critical_temp']:.2f} K")
                st.metric("Temp (K)", f"{result['critical_temp']:.2f}")
            else:
                st.error(f"API Error: {response.text}")

elif input_mode == "Batch Upload (CSV)":
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV (must contain 81 feature columns)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", df.head())
        features_df = df.iloc[:, :81]
        
        if st.button("Predict All"):
            results = []
            bar = st.progress(0)
            
            for i, (index, row) in enumerate(features_df.iterrows()):
                # Assuming the CSV has exactly the 81 features in order
                payload = {"data": row.tolist()}
                resp = requests.post(API_URL, json=payload)
                if resp.status_code == 200:
                    results.append(resp.json()['critical_temp'])
                bar.progress((i + 1) / len(df))
            
            df['Predicted_Critical_Temp'] = results
            st.write("Results:", df)
            st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")