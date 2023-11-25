


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function to preprocess the data and train the Random Forest model
def train_random_forest(data):
    # Encode crop names using LabelEncoder
    le = LabelEncoder()
    data['crop_names'] = le.fit_transform(data['crop_names'])

    # Split the data into features and target variable
    X = data.drop('production', axis=1)
    y = data['production']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate and print the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
   # st.write(f'Mean Squared Error: {mse}')

    return rf, le

# Function to predict crop yield for user input
def predict_yield(model, le, crop_names, humidity, wind_speed, area, temperature, precipitation, N, P, K):
    crop_name_encoded = le.transform([crop_names])[0]
    input_data = np.array([[crop_name_encoded, area,temperature,wind_speed,precipitation,humidity, N, P, K]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit web application
def main():
    st.title('Crop Yield Prediction')

    # Upload CSV file
    data = pd.read_csv("crop_yeild.csv")

    # Train the Random Forest model
    model, le = train_random_forest(data)

    # User input for prediction
    st.sidebar.header('Enter Input Features:')
    crop_names = st.option = st.selectbox(
    'Crop Name',
    ('Rice', 'Banana', 'Maize', 'Cotton(lint)', 'Ragi', 'Onion',
       'Potato', 'Tomato', 'Carrot', 'Drum Stick'))

    area = st.number_input('Area (hat)', min_value=0.0, max_value=687000.0, value=100.0)
    temperature = st.number_input('Temperature (Celsius)', min_value=0.0, max_value=25.0, value=25.0)
    precipitation = st.number_input('Precipitation (mm)', min_value=0.0, max_value=1100.0, value=50.0)
    wind_speed = st.number_input('Wind Speed (mm)', min_value=0.0, max_value=4.0, value=1.0)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
    N = st.number_input('Nitrogen (kg/ha)', min_value=0.0, max_value=1000.0, value=50.0)
    P = st.number_input('Phosphorus (kg/ha)', min_value=0.0, max_value=120.0, value=20.0)
    K = st.number_input('Potassium (kg/ha)', min_value=0.0, max_value=60.0, value=30.0)

    # Predict and display the result
    if st.button('Predict'):
        prediction = predict_yield(model, le, crop_names, humidity, wind_speed, area, temperature, precipitation, N, P, K)
        

        st.subheader(f'Predicted Yield: {(prediction)} tons/hat')

if __name__ == '__main__':
    main()
