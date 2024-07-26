# Medical Cost Prediction

## Overview

This project involves predicting medical costs for individuals using a Linear Regression model. The preprocessing steps include One-Hot Encoding for categorical variables and Standard Scaling for numerical features. The performance of the model is evaluated using the R² score.

## Dataset

This dataset contains detailed information about medical costs for individuals over the period from 2010 to 2020. It includes various attributes such as:

- **Age**: Age of the individual
- **Sex**: Gender of the individual 
- **BMI**: Body Mass Index of the individual
- **Children**: Number of children/dependents of the individual
- **Smoker**: Smoking status 
- **Region**: Geographical region ('northwest', 'northeast', 'southeast', 'southwest')
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
    - Sex (male or female)
    - BMI (numeric)
    - Children (numeric)
    - Smoker ( No or Yes)
    - Region (one of 'northwest', 'northeast', 'southeast', 'southwest')

3. **Make Predictions**: Use the loaded model to predict medical costs based on the input data.


