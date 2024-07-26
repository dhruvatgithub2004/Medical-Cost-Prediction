# Medical Cost Prediction

## Overview

This project involves predicting medical costs for individuals using a Linear Regression model. The preprocessing steps include One-Hot Encoding for categorical variables and Standard Scaling for numerical features. The performance of the model is evaluated using the R² score.

## Dataset

This dataset contains detailed information about medical costs for individuals over the period from 2010 to 2020. It includes various attributes such as:

- **Age**: Age of the individual
- **Sex**: Gender of the individual (0: Male, 1: Female)
- **BMI**: Body Mass Index of the individual
- **Children**: Number of children/dependents of the individual
- **Smoker**: Smoking status (0: No, 1: Yes)
- **Region**: Geographical region (e.g., 'northwest', 'northeast', 'southeast', 'southwest')
- **Medical Cost**: Medical cost incurred by the individual

## Model

### Preprocessing

1. **One-Hot Encoding**: Applied to categorical columns `Sex`, `Smoker`, and `Region` to convert them into numerical format.
2. **Standard Scaling**: Applied to numerical columns `Age`, `BMI`, and `Children` to standardize their values.

### Model

- **Linear Regression**: Used to predict the medical costs based on the processed features.

### Evaluation

- **R² Score**: The model achieved an R² score of 99%, indicating that 99% of the variance in medical costs is explained by the model.

## Usage

To use the model for predictions, follow these steps:

1. **Load the Model**: The model is saved in a pickle file (`medical_cost_prediction.pkl`).

2. **Input Data**: Provide the input features in the following format:
    - Age (numeric)
    - Sex (0 for Male, 1 for Female)
    - BMI (numeric)
    - Children (numeric)
    - Smoker (0 for No, 1 for Yes)
    - Region (one of 'northwest', 'northeast', 'southeast', 'southwest')

3. **Make Predictions**: Use the loaded model to predict medical costs based on the input data.

## Example

Here is an example of how to load the model and make predictions using Streamlit:

```python
import pickle
import pandas as pd
import streamlit as st

# Load the model
with open('medical_cost_prediction.pkl', "rb") as file:
    pipeline = pickle.load(file)

# Streamlit application
st.title("Medical Cost Prediction")

age = st.number_input('Age', min_value=0, max_value=120, value=25)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
bmi = st.number_input('BMI', min_value=0, max_value=40, value=25)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
region = st.selectbox('Region', options=['northwest', 'northeast', 'southeast', 'southwest'])

# Create DataFrame
input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                          columns=['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region'])

# Predict
if st.button('Predict'):
    prediction = pipeline.predict(input_data)
    st.write('Prediction:', prediction[0])
