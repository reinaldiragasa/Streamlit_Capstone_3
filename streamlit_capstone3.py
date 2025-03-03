# Import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Utama
st.title('Housing Price Predictor')
st.text('This web can be used to predict housing prices')

# Menambahkan sidebar
st.sidebar.header("Please input the housing features")

def create_user_input():
    # Numerical Features
    longitude = st.sidebar.number_input('Longitude', min_value=-124.35, max_value=-114.31, value=-119.0, step=0.01)
    latitude = st.sidebar.number_input('Latitude', min_value=32.54, max_value=41.95, value=37.0, step=0.01)
    housing_median_age = st.sidebar.number_input('Housing Median Age', min_value=1, max_value=52, value=30)
    total_rooms = st.sidebar.number_input('Total Rooms', min_value=2, max_value=32627, value=2000)
    total_bedrooms = st.sidebar.number_input('Total Bedrooms', min_value=2, max_value=6445, value=400)
    population = st.sidebar.number_input('Population', min_value=3, max_value=35682, value=3000)
    households = st.sidebar.number_input('Households', min_value=2, max_value=6082, value=800)
    median_income = st.sidebar.number_input('Median Income', min_value=499.9, max_value=150001.0, value=50000.0, step=1.0)
    rooms_per_household = st.sidebar.number_input('Rooms per Household', min_value=2.059769, max_value=8.440329, value=3.0, step=0.01)
    bedrooms_per_room = st.sidebar.number_input('Bedrooms per Room', min_value=0.079867, max_value=0.335025, value=0.1, step=0.01)
    population_per_household = st.sidebar.number_input('Population per Household', min_value=1.147406, max_value=4.560277, value=2.0, step=0.01)
    income_per_household = st.sidebar.number_input('Income per Household', min_value=0.379596, max_value=27.629386, value=5.0, step=0.01)
    price_per_room = st.sidebar.number_input('Price per Room', min_value=2.255639, max_value=274.845958, value=50.0, step=0.01)
    rooms_per_person = st.sidebar.number_input('Rooms per Person', min_value=0.372505, max_value=3.449184, value=1.0, step=0.01)
    bedrooms_per_person = st.sidebar.number_input('Bedrooms per Person', min_value=0.124789, max_value=0.634712, value=0.2, step=0.01)

    # Categorical Features
    ocean_proximity = st.sidebar.radio('Ocean Proximity', ['<1H_OCEAN', 'INLAND', 'NEAR OCEAN','NEAR BAY'])
    age_category = st.sidebar.radio('Age Category', ['Medium', 'Old', 'New'])
    income_category = st.sidebar.radio('Income Category', ['Medium', 'Low', 'High'])
    ocean_proximity_category = st.sidebar.radio('Ocean Proximity Category', ['Coastal', 'Inland'])
    house_size_category = st.sidebar.radio('House Size Category', ['Medium', 'Small', 'Large'])
    price_category = st.sidebar.radio('Price Category', ['Medium', 'Low', 'High'])
    is_coastal = st.sidebar.radio('Coastal Category', ['COASTAL', 'NOT COASTAL'])
    is_urban = st.sidebar.radio('Urban Category', ['Urban', 'Not Urban'])
    is_affluent = st.sidebar.radio('Affluent Category', ['Not Affluent', 'Affluent'])
    is_luxury = st.sidebar.radio('Luxury Category', ['Not Luxury', 'Luxury'])
    is_high_density = st.sidebar.radio('High Density Category', ['Not High Density', 'High Density '])
    city = st.sidebar.radio('City', ['Los Angeles', 'San Francisco', 'Fresno', 'San Diego', 'Sacramento', 'Riverside', 'Bakersfield', 'Redding', 'Eureka', 'Yreka'])

    # Creating a dictionary with user input
    user_data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household,
        'income_per_household': income_per_household,
        'price_per_room': price_per_room,
        'rooms_per_person': rooms_per_person,
        'bedrooms_per_person': bedrooms_per_person,
        'ocean_proximity': ocean_proximity,
        'age_category': age_category,
        'income_category': income_category,
        'ocean_proximity_category': ocean_proximity_category,
        'house_size_category': house_size_category,
        'price_category': price_category,
        'is_coastal': is_coastal,
        'is_urban': is_urban,
        'is_affluent': is_affluent,
        'is_luxury': is_luxury,
        'is_high_density': is_high_density,
        'city': city
    }
    
    # Convert dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat 2 kolom
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Housing Features")
    st.write(data_customer.transpose())

# Load model
with open('best_model_cat.sav', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
predicted_price = model_loaded.predict(data_customer)

# Menampilkan hasil prediksi
with col2:
    st.subheader('Prediction Result')
    st.write(f'Predicted House Price: ${predicted_price[0]:,.2f}')
