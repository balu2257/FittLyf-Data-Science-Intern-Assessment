# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:03:07 2024

@author: balak
"""

import streamlit as st
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
import numpy as np

# Define a function to load data and predict anomalies
def load_and_predict(file):
    # Load the data
    data = pd.read_csv(file)
    
    # Check if necessary columns are present
    required_columns = ['Amount', 'Time', 'Class']
    if not all(col in data.columns for col in required_columns):
        st.error("Dataset must contain 'Amount', 'Time', and 'Class' columns.")
        return None

    # Prepare features for prediction
    features = data[['Amount', 'Time']]
    
    # Initialize models
    iso_forest = IForest(contamination=0.001)
    lof = LOF(n_neighbors=20, contamination=0.001)
    
    # Fit models
    iso_forest.fit(features)
    lof.fit(features)
    
    # Predict anomalies
    iso_preds = iso_forest.predict(features)
    lof_preds = lof.predict(features)
    
    # Add predictions to data
    data['Isolation_Forest'] = iso_preds
    data['LOF'] = lof_preds
    data['Isolation_Forest'] = data['Isolation_Forest'].astype(int)
    data['LOF'] = data['LOF'].astype(int)
    
    return data

# Streamlit app
st.title('Credit Card Fraud Detection')

uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    result = load_and_predict(uploaded_file)
    
    if result is not None:
        st.write(result.head())  # Display first few rows of the dataframe
        st.download_button("Download Results", result.to_csv(index=False), "results.csv")

        # Visualizations for anomalies
        st.subheader('Isolation Forest Anomalies')
        st.write(result[result['Isolation_Forest'] == 1].head())
        
        st.subheader('LOF Anomalies')
        st.write(result[result['LOF'] == 1].head())
