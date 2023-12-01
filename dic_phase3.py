import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pickled model
with open('catboost_model.pkl', 'rb') as file:
    catboost_regressor = pickle.load(file)

# Function to preprocess user input
def preprocess_input(sales_price, base_price, featured_item, prominently_displayed, scaler):
    input_data = pd.DataFrame({
        'Sales_Price': [sales_price],
        'Base_Price': [base_price],
        'Featured_Item_Of_Week_Featured': [featured_item],
        'Displayed_Prominently_On Display': [prominently_displayed]
    })

    # Normalize input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

# Streamlit app
def main():
    st.title("Sales Prediction App")

    # Collect user input
    sales_price = st.number_input("Enter Sales Price:")
    base_price = st.number_input("Enter Base Price:")
    featured_item = st.checkbox("Featured Item of the Week (1 for Yes, 0 for No)")
    prominently_displayed = st.checkbox("Displayed Prominently (1 for Yes, 0 for No)")

    if st.button("Predict"):
        # Load the scaler used for training
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Preprocess user input
        input_data_scaled = preprocess_input(sales_price, base_price, featured_item, prominently_displayed, scaler)

        # Make prediction
        prediction = catboost_regressor.predict(input_data_scaled)

        # Display result
        st.success(f"Predicted Number of Units Sold: {prediction}")

if __name__ == "__main__":
    main()
