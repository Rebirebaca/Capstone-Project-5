import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu


# Load the model
with open("C:\\Users\\jenis\\Desktop\\project_files\\decisiontree.pkl",'rb') as f:
    model = pickle.load(f)

# Feature names:
list_of_features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model',
            'year', 'month_of_year', 'lease_commence_year',
            'remaining_lease_years', 'remaining_lease_months']

# Categorical variable mappings:
encoded_features = {
    'town': {'SENGKANG': 20, 'PUNGGOL': 17, 'WOODLANDS': 24, 'YISHUN': 25,
            'TAMPINES': 22, 'JURONG WEST': 13, 'BEDOK': 1, 'HOUGANG': 11,
            'CHOA CHU KANG': 8, 'ANG MO KIO': 0, 'BUKIT MERAH': 4, 'BUKIT PANJANG': 5,
            'BUKIT BATOK': 3, 'TOA PAYOH': 23, 'PASIR RIS': 16, 'KALLANG/WHAMPOA': 14,
            'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'GEYLANG': 10, 'CLEMENTI': 9,
            'JURONG EAST': 12, 'BISHAN': 2, 'SERANGOON': 21, 'CENTRAL AREA': 7,
            'MARINE PARADE': 15, 'BUKIT TIMAH': 6},
    
    'flat_type': {'4 ROOM': 3, '5 ROOM': 4, '3 ROOM': 2, 'EXECUTIVE': 5, '2 ROOM': 1, 'MULTI-GENERATION': 6, '1 ROOM': 0},
    
    'storey_range': {'04 TO 06': 1, '07 TO 09': 2, '10 TO 12': 3, '01 TO 03': 0,
                    '13 TO 15': 4, '16 TO 18': 5, '19 TO 21': 6, '22 TO 24': 7,
                    '25 TO 27': 8, '28 TO 30': 9, '31 TO 33': 10, '34 TO 36': 11,
                    '37 TO 39': 12, '40 TO 42': 13, '43 TO 45': 14, '46 TO 48': 15,
                    '49 TO 51': 16},
    
    'flat_model': {'Model A': 8, 'Improved': 5, 'New Generation': 12, 'Premium Apartment': 13,
                'Simplified': 16, 'Apartment': 3, 'Maisonette': 7, 'Standard': 17,
                'DBSS': 4, 'Model A2': 10, 'Model A-Maisonette': 9, 'Adjoined flat': 2,
                'Type S1': 19, 'Type S2': 20, 'Premium Apartment Loft': 14, 'Terrace': 18,
                'Multi Generation': 11, '2-room': 0, 'Improved-Maisonette': 6, '3Gen': 1,
                'Premium Maisonette': 15},}

# Input widgets for user interaction
st.set_page_config(layout="wide")

st.write("""<div style='text-align:center'> <h1 style='color:#e377c2;'>Singapore Resale Flat Prices Prediction</h1> </div>""", unsafe_allow_html=True)

select = option_menu(None, ['HOME','About Project','PRICE PREDICTION'],
                        icons=['house','gear','cash-coin'],orientation='vertical',default_index=0)

if select =='HOME':
        st.write('# WELCOME TO SINGAPORE FLAT PRICE PREDICTION')
        st.markdown('#### This project aims to construct a machine learning model and implement it as a user-friendly online application in order to provide accurate predictions about the resale values of apartments in Singapore.')
        st.write('### TECHNOLOGY USED')
        st.write('- PYTHON   (PANDAS, NUMPY)')
        st.write('- SCIKIT-LEARN')
        st.write('- DATA PREPROCESSING')
        st.write('- EXPLORATORY DATA ANALYSIS')
        st.write('- PICKLE')
        st.write('- STREAMLIT')
        st.write('### MACHINE LEARNING MODEL')
        st.write('#### REGRESSION - Random Forest Regressor')
        st.write('- The Random Forest Regressor is a type of ensemble learning algorithm that belongs to the family of decision tree algorithms.')


if select == 'About Project':
    st.write('### In this project, The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat.')
    st.write('### There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration.')
    st.write('### A predictive model can help to overcome these challenges by providing users with an estimated resale price based on these factors.')
    st.write('### The objective is to develop a Streamlit webpage that enables users to input values for each column and get the expected resale_price value for the flats in Singapore.')

if select=='PRICE PREDICTION':
    int_data = {}
    for i in list_of_features:
        if i in encoded_features:
            selected_option = st.sidebar.selectbox(f"Select {i.capitalize()}:", options=list(encoded_features[i].keys()))
            int_data[i] = encoded_features[i][selected_option]

            int_data[i] = st.sidebar.number_input(f"{i.capitalize()}:")
        else:
            int_data[i] = st.sidebar.number_input(f"{i.capitalize()}:")

# Make predictions using the loaded model
    if st.button("Predict"):
        int_array = np.array([int_data[i] for i in list_of_features]).reshape(1, -1)
        prediction = model.predict(int_array)

    # Display the prediction result
        prediction_scale = np.exp(prediction[0])
        st.markdown("<h3 style='color:#007A74;'>Prediction Result:</h3>", unsafe_allow_html=True)
        st.write(f"<span style='color:#009999;'>The predicted house price is: {prediction_scale:,.2f} INR</span>", unsafe_allow_html=True)
