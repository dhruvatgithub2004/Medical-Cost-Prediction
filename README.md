# Medical Cost Prediction Project

**Team Name:** Individual Project

## Problem Statement

The project addresses the complex challenge of **predicting medical costs**. Healthcare expenses can vary significantly based on individual characteristics and behaviours, making accurate cost estimation difficult. This project aims to develop an AI-powered solution to forecast these costs, which can be beneficial for various stakeholders in the healthcare sector, such as insurance companies, healthcare providers, and policymakers, for planning and resource allocation.

## Solution Overview

This project presents an AI-powered solution that leverages machine learning models to **predict individual medical costs**. The solution involves comprehensive data preprocessing, including handling categorical features and scaling numerical data. Multiple regression models, including Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor, are trained and evaluated to identify the most accurate predictor. The trained model can then be used to estimate medical expenses based on a patient's characteristics.

## AI/Tech Stack

The project utilises a variety of Python libraries and machine learning models for data handling, analysis, and prediction:

*   **Programming Language:** Python
*   **Data Manipulation:** `pandas`
*   **Data Visualisation:** `matplotlib.pyplot`, `seaborn` 
*   **Machine Learning Framework:** `scikit-learn` (sklearn)
    *   **Data Preprocessing:**
        *   `ColumnTransformer`: For applying different transformations to different columns.
        *   `OneHotEncoder`: To convert categorical features (`Sex`, `Smoker`, `Region`) into numerical format.
        *   `StandardScaler`: To scale numerical features (`Age`, `BMI`, `Children`).
    *   **Model Selection & Evaluation:**
        *   `train_test_split`: For splitting data into training and testing sets.
        *   `cross_val_score`: For robust model evaluation using cross-validation.
        *   `mean_squared_error`, `r2_score`, `mean_absolute_error`: For evaluating model performance.
    *   **Machine Learning Models:**
        *   `LinearRegression` 
        *   `RandomForestRegressor`
        *   `GradientBoostingRegressor`
    *   **Pipeline Management:** `Pipeline` 
*   **Model Persistence:** `pickle` 

## Installation & Setup Instructions

To run this project, follow these steps:

1.  **Clone the repository:** (Assumed, as the code is provided in excerpts)
    ```bash
    git clone https://github.com/dhruvatgithub2004/Medical-Cost-Prediction.git
    ```
2.  **Ensure Python is installed:** Python 3.10 is recommended.
3.  **Install necessary libraries:** You can install all required libraries using `pip`.
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```
    *(Note: `numpy` is implicitly a dependency of `scikit-learn` and `pandas` and will be installed automatically).*
4.  **Download the dataset:** The project reads data from a CSV file named `medical cost regression.csv`. Ensure this file is located at the specified path (`C:\Users\dkdes\OneDrive\Desktop\kaggle_datasets\medical cost regression.csv`) or update the `pd.read_csv` line in the script.
5.  **Run the Python script(s):** Execute the main script containing the data loading, preprocessing, model training, and evaluation steps.
    ```bash
    python medical_cost_prediction_app.py
    ```
    *(The provided sources are code excerpts, implying they are part of a larger script or notebook.)*
6.  **Load the pre-trained model (Optional):** If you wish to use the saved pipeline, ensure `medical Cost prediction.pkl` is available in your project directory and load it using `pickle`.

## Key Features

The solution encompasses the following functionalities:

*   **Data Loading and Initial Exploration:** Loads medical cost data from a CSV file and provides an initial overview of its structure, data types, and non-null counts.
*   **Exploratory Data Analysis (EDA):** Visualises the distribution of numerical features (e.g., Age, BMI, Children) and relationships between variables (e.g., average medical cost by Smoker status, Region, Sex, Age).
*   **Data Preprocessing:**
    *   Handles categorical features (`Sex`, `Smoker`, `Region`) using One-Hot Encoding.
    *   Scales numerical features (`Age`, `BMI`, `Children`) using Standard Scaling.
*   **Train-Test Split:** Divides the dataset into training and testing sets to evaluate model performance on unseen data.
*   **Multiple Model Training:** Implements and trains different regression models, including:
    *   Linear Regression
    *   Random Forest Regressor
    *   Gradient Boosting Regressor
*   **Model Evaluation:** Assesses model performance using metrics such as R² score and Mean Absolute Error (MAE). The Gradient Boosting model achieved an R² Score of **0.9976575465426418** on test data.
*   **Overfitting Check:** Utilises cross-validation to assess if models are overfitting to the training data.
*   **Feature Importance Analysis:** Identifies the most important features contributing to medical cost prediction using the Random Forest model.
*   **Model Persistence:** Saves the trained machine learning pipeline using `pickle` for future use or deployment.

## Global/Regional Adaptability Statement

The current solution is trained on a dataset that includes a 'Region' feature. While the model inherently learns patterns from this regional data, its direct applicability to different countries or vastly different healthcare systems without retraining might be limited.

To adapt or scale this solution globally or for distinct healthcare systems:

*   **New Data Collection:** The model would need to be re-trained on new datasets that reflect the specific demographics, healthcare policies, economic factors, and medical cost structures of the target country or region.
*   **Feature Engineering:** Additional relevant features unique to different healthcare systems (e.g., type of insurance coverage, specific local health regulations, prevalence of certain diseases in a region) might need to be incorporated.
*   **Regulatory Compliance:** Any deployment would need to comply with local data privacy regulations (e.g., GDPR, HIPAA) and healthcare laws.
*   **Cultural Context:** Factors influencing health behaviours and access to care can vary significantly and should be considered in data collection and model interpretation.

## 🧠 Ethical Considerations

- **No Data Storage:** User inputs are not stored, logged, or shared. All predictions are processed within the session to protect user privacy.
- **No Personal Identifiers Collected:** The app does not request or handle any personally identifiable health information (PII/PHI).
- **Bias Awareness:** The machine learning model is trained on anonymized, balanced data to reduce potential bias across demographic and lifestyle factors.
- **Explainable AI:** A simple, transparent linear regression model is used to ensure predictions are understandable and trustworthy.


## Future Enhancements

Potential improvements and additional features for this project include:

*   **Hyperparameter Tuning:** Conducting extensive hyperparameter optimisation for the chosen models (Random Forest, Gradient Boosting) to achieve peak performance.
*   **Insurance plan optimization**
*   **mobile responsiveness**
*   **EHR or insurance API integration**
*   **mulltilingual support**
*   **Uncertainty Quantification:** Providing not just a point prediction for medical cost, but also a confidence interval or range, to give a better sense of prediction uncertainty.
*   **Interactive Application:** Developing a user-friendly web application or interface to allow users to input patient details and receive instant medical cost predictions.
*   **Continuous Learning:** Implementing a system for continuous model retraining and updates with new data to maintain accuracy as healthcare trends evolve.

## Link to the app
https://medical-cost-prediction-kaz789d6pcpqnpdpabbsst.streamlit.app/

## Team Members

*   **Dhruv Desai** - Individual Contributor
