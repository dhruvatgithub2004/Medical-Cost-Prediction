import pickle
import streamlit as st
import pandas as pd

# Load the pre-trained pipeline
with open('medical Cost prediction.pkl', "rb") as file:
    pipeline = pickle.load(file)

# Streamlit app title
st.title("Medical Cost Prediction")

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=25)
gender = st.selectbox('Gender', options=['male', 'female'])
BMI = st.number_input("BMI", min_value=0.0, max_value=40.0, value=15.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", options=['yes', 'no'])
region = st.selectbox("Region", options=['northwest', 'northeast', 'southeast', 'southwest'])

# Create input DataFrame
input_data = pd.DataFrame([[age, gender, BMI, children, smoker, region]],
                          columns=['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region'])

# Predict button
if st.button('Predict'):
    try:
        # Predict using the pipeline
        prediction = pipeline.predict(input_data)
        st.write('Prediction:', prediction[0])
    except Exception as e:
        st.error(f"An error occurred: {e}")
