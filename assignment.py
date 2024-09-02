# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:03:07 2024

@author: balak
"""

 import streamlit as st
import pandas as pd
import numpy as np

# Define a function to load data and detect anomalies
def load_and_predict(file, file_type):
    # Load the data based on file type
    if file_type == 'csv':
        data = pd.read_csv(file)
    elif file_type == 'excel':
        data = pd.read_excel(file)
    
    # Check if necessary columns are present
    required_columns = ['Amount', 'Time', 'Class']
    if not all(col in data.columns for col in required_columns):
        st.error("Dataset must contain 'Amount', 'Time', and 'Class' columns.")
        return None

    # Simple statistical anomaly detection
    mean_amount = data['Amount'].mean()
    std_amount = data['Amount'].std()
    
    # Define a threshold for anomaly detection
    threshold = mean_amount + 3 * std_amount
    
    # Detect anomalies
    data['Anomaly'] = data['Amount'].apply(lambda x: 1 if x > threshold else 0)
    
    return data

# Streamlit app
st.title('Credit Card Fraud Detection')

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Determine file type
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'csv':
        file_type = 'csv'
    elif file_type == 'xlsx':
        file_type = 'excel'
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        file_type = None
    
    if file_type:
        result = load_and_predict(uploaded_file, file_type)
        
        if result is not None:
            st.write(result.head())  # Display first few rows of the dataframe
            st.download_button("Download Results", result.to_csv(index=False) if file_type == 'csv' else result.to_excel(index=False), "results." + file_type)
            
            # Visualization
            st.subheader('Anomalies')
            st.write(result[result['Anomaly'] == 1].head())
            
            # Plotting transaction amounts
            st.subheader('Transaction Amount Distribution')
            st.bar_chart(result['Amount'])
