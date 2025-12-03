import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# -------------------------
# Load models safely
# -------------------------
try:
    # Regression & preprocessing
    reg_pipe = joblib.load('models/RandomForest_pipe.pkl')
    scaler = joblib.load('models/scaler.pkl')

    # Clustering
    cluster_model = joblib.load('models/cluster_model.pkl')
    cluster_scaler = joblib.load('models/cluster_scaler.pkl')
    pca = joblib.load('models/pca.pkl')

    # ANN model
    try:
        ann = tf.keras.models.load_model('models/ann_regressor.h5')
    except:
        ann = None
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# -------------------------
# USD to INR conversion
# -------------------------
USD_TO_INR = 83.0  # Update dynamically if needed

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Diamond Dynamics", layout="centered")
st.title("💎 Diamond Dynamics: Price Prediction & Market Segmentation")

# -------------------------
# Input Form
# -------------------------
with st.form("input_form"):
    st.subheader("Enter Diamond Attributes")
    carat = st.number_input("Carat", min_value=0.01, max_value=10.0, value=0.5, step=0.01)
    cut = st.selectbox("Cut", ['Fair','Good','Very Good','Premium','Ideal'])
    color = st.selectbox("Color", ['D','E','F','G','H','I','J'])
    clarity = st.selectbox("Clarity", ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])
    depth = st.number_input("Depth (%)", value=61.5)
    table = st.number_input("Table (%)", value=57.0)
    x = st.number_input("Length (mm)", value=5.2)
    y = st.number_input("Width (mm)", value=5.1)
    z = st.number_input("Depth (mm)", value=3.2)
    submitted = st.form_submit_button("Predict")

# -------------------------
# Prediction Section
# -------------------------
if submitted:
    # -------------------------
    # Encode ordinals (same as training)
    # -------------------------
    cut_map = {'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5}
    color_map = {'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7}
    clarity_map = {'I1':1,'SI2':2,'SI1':3,'VS2':4,'VS1':5,'VVS2':6,'VVS1':7,'IF':8}

    cut_ord = cut_map[cut]
    color_ord = color_map[color]
    clarity_ord = clarity_map[clarity]

    # Derived features
    volume = x * y * z
    input_df = pd.DataFrame([{
        'carat': carat,
        'cut_ord': cut_ord,
        'color_ord': color_ord,
        'clarity_ord': clarity_ord,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z,
        'volume': volume
    }])

    # -------------------------
    # Price Prediction (Random Forest)
    # -------------------------
    try:
        pred_price = reg_pipe.predict(input_df)[0]
    except Exception as e:
        st.error(f"Error predicting price: {e}")
        pred_price = 0

    # -------------------------
    # ANN Price Prediction (optional)
    # -------------------------
    pred_price_ann = None
    if ann is not None:
        try:
            ann_input = scaler.transform(input_df)
            pred_price_ann = ann.predict(ann_input)[0][0]
        except Exception as e:
            pred_price_ann = None

    # -------------------------
    # Display Price
    # -------------------------
    price_inr = pred_price * USD_TO_INR
    st.metric("Predicted Price (USD)", f"${pred_price:,.2f}")
    st.metric("Predicted Price (INR)", f"₹{price_inr:,.2f}")

    if pred_price_ann is not None:
        st.write(f"ANN predicted price (USD): ${pred_price_ann:,.2f}")

    # -------------------------
    # Cluster Prediction
    # -------------------------
    try:
        cl_input = input_df.copy()
        cl_scaled = cluster_scaler.transform(cl_input)
        cl_label = cluster_model.predict(cl_scaled)[0]

        # Map cluster to name (update mapping as per your cluster analysis)
        cluster_names = {0:'Premium Heavy Diamonds',
                         1:'Affordable Small Diamonds',
                         2:'Mid-range Balanced Diamonds'}
        cluster_name = cluster_names.get(cl_label, f"Cluster {cl_label}")
        st.success(f"Cluster: {cl_label} — {cluster_name}")

        # PCA coordinates for visualization
        pc = pca.transform(cl_scaled)
        st.write(f"PCA coordinates: PC1={pc[0,0]:.3f}, PC2={pc[0,1]:.3f}")
    except Exception as e:
        st.error(f"Error predicting cluster: {e}")
