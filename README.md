# Customer Churn Prediction using ANN

This project is a machine learning web application built with Streamlit that predicts whether a bank customer will churn (leave the bank) or not based on various customer demographics and account information.

The prediction model uses an Artificial Neural Network (ANN) built with TensorFlow/Keras.

## Features

- **Interactive Web UI:** A user-friendly Streamlit interface to input customer details.
- **Real-time Prediction:** Instantly predicts the probability of a customer churning.
- **Data Preprocessing:** Handles categorical data encoding (Label Encoding for Gender, One-Hot Encoding for Geography) and feature scaling.
- **Deep Learning Model:** Uses a trained Artificial Neural Network (ANN) saved as an `.h5` file.

## Technologies Used

- **Python 3.13**
- **Streamlit:** For the web application frontend and backend.
- **TensorFlow & Keras:** For the Artificial Neural Network model.
- **Scikit-Learn:** For data preprocessing (StandardScaler, LabelEncoder, OneHotEncoder).
- **Pandas & NumPy:** For data manipulation and arrays.

## Installation and Setup

Clone the repository (if applicable) or navigate to the project directory:

```bash
cd "Customer-Churn Prediction"
```

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

**On Windows:**

```bash
.\venv\Scripts\activate
```

**On macOS/Linux:**

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, you can install the main packages using:

```bash
pip install streamlit numpy pandas scikit-learn tensorflow
```

## Usage

To run the Streamlit application locally, run the following command in your terminal while the virtual environment is active:

```bash
streamlit run app.py
```

This starts a local web server and opens the application in your default web browser (usually at `http://localhost:8501`).

## Project Files

| File | Description |
|------|-------------|
| `app.py` | The main Streamlit web application script. |
| `model.h5` | The trained Artificial Neural Network model. |
| `scaler.pkl` | The saved StandardScaler object used to normalize numerical input features. |
| `label_encoder_gender.pkl` | The saved LabelEncoder object for the `Gender` column. |
| `onehot_encoder_geo.pkl` | The saved OneHotEncoder object for the `Geography` column. |
| `hyperparametertuningann.ipynb` | Jupyter notebook containing the code used for hyperparameter tuning and model training. |

## Input Parameters

The application takes the following customer details to make a prediction:

| Parameter | Description |
|-----------|-------------|
| **Geography** | Country of the customer (e.g., France, Spain, Germany). |
| **Gender** | Male or Female. |
| **Age** | Age of the customer. |
| **Balance** | Current account balance. |
| **Credit Score** | Customer's credit score. |
| **Estimated Salary** | Customer's estimated annual salary. |
| **Tenure** | Number of years the customer has been with the bank. |
| **Number of Products** | Number of bank products the customer uses. |
| **Has Credit Card** | Whether the customer has a credit card (1 = Yes, 0 = No). |
| **Is Active Member** | Whether the customer is an active member (1 = Yes, 0 = No). |

**Prediction rule:** The model outputs a churn probability between 0 and 1. Values above 0.5 are interpreted as likely to churn; values at or below 0.5 as unlikely to churn.
