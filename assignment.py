# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:03:07 2024

@author: balak
"""
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# Pre-trained models (assuming they were already trained before deployment)
iso_forest = IsolationForest(contamination=0.001, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001, novelty=True)

# Function to load data and predict anomalies
def load_and_predict(file):
    try:
        data = pd.read_csv(file)
        
        if not {'Amount', 'Time'}.issubset(data.columns):
            st.error("Uploaded file must contain 'Amount' and 'Time' columns.")
            return None
        
        # Dropping unnecessary columns if needed
        features = data.drop(['Class'], axis=1, errors='ignore')
        
        # Predicting anomalies using Isolation Forest and Local Outlier Factor
        iso_preds = iso_forest.fit_predict(features)
        lof_preds = lof.fit_predict(features)
        
        # Converting predictions to binary (0: normal, 1: anomaly)
        data['Isolation_Forest'] = pd.Series(iso_preds).map({1: 0, -1: 1})
        data['LOF'] = pd.Series(lof_preds).map({1: 0, -1: 1})
        
        return data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app interface
st.title('Credit Card Fraud Detection')

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    result = load_and_predict(uploaded_file)
    
    if result is not None:
        st.write(result.head())  # Display the top rows of the result
        
        # Visualization: Show the number of anomalies detected
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        result['Isolation_Forest'].value_counts().plot(kind='bar', ax=ax[0], color=['blue', 'red'])
        ax[0].set_title('Isolation Forest Anomaly Counts')
        ax[0].set_xlabel('Class')
        ax[0].set_ylabel('Count')
        ax[0].set_xticklabels(['Normal', 'Anomaly'], rotation=0)
        
        result['LOF'].value_counts().plot(kind='bar', ax=ax[1], color=['blue', 'red'])
        ax[1].set_title('LOF Anomaly Counts')
        ax[1].set_xlabel('Class')
        ax[1].set_ylabel('Count')
        ax[1].set_xticklabels(['Normal', 'Anomaly'], rotation=0)
        
        st.pyplot(fig)
        
        # Provide an option to download the results
        st.download_button("Download Results", result.to_csv(index=False), "results.csv")