import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load the model
model = joblib.load("model.pkl")

st.title("My ML Model Predictor")

# Create inputs for your features
area = st.number_input("Enter surface area of the house")
bedrooms = st.number_input("Enter number of bedrooms")
bathrooms = st.number_input("Enter number of bathrooms")
stories = st.number_input("Enter number of stories")
parking = st.number_input("Enter number of parking spaces")
furnishingstatus_semi_furnished = st.selectbox("Is the house semi-furnished?", [0, 1])
furnishingstatus_unfurnished = st.selectbox("Is the house unfurnished?", [0, 1])
mainroad_yes = st.selectbox("Is the house on the main road?", [0, 1])
airconditioning_yes = st.selectbox("Does the house have air conditioning?", [0, 1])
prefarea_yes = st.selectbox("Is the house in a preferred area?", [0, 1])
basement_yes = st.selectbox("Does the house have a basement?", [0, 1])
guestroom_yes = st.selectbox("Does the house have a guest room?", [0, 1])
hotwaterheating_yes = st.selectbox("Does the house have hot water heating?", [0, 1])

# Calculate derived features
amenities = [furnishingstatus_semi_furnished, furnishingstatus_unfurnished, mainroad_yes, airconditioning_yes, prefarea_yes, basement_yes, guestroom_yes, hotwaterheating_yes]
amenity_score = sum(amenities)

# applying the same transformations as in training, standatd scaler to features
X = np.array([[area, bedrooms, bathrooms, stories, parking, furnishingstatus_semi_furnished, furnishingstatus_unfurnished, mainroad_yes, airconditioning_yes, prefarea_yes, basement_yes, guestroom_yes, hotwaterheating_yes, amenity_score]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 


if st.button("Predict"):
    prediction_log = model.predict(X_scaled)
    prediction = np.expm1(prediction_log)  # Inverse of log1p transformation
    st.write(f"Prediction: {prediction[0]}")