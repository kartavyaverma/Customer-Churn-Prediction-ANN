# Customer Churn Prediction using ANN

This project is a machine learning web application built with Streamlit that predicts whether a bank customer will churn (leave the bank) or not based on various customer demographics and account information.

The prediction model uses an Artificial Neural Network (ANN) built with TensorFlow/Keras.

## Features

- **Interactive Web UI:** A user-friendly Streamlit interface to input customer details.
- **Real-time Prediction:** Instantly predicts the probability of a customer churning.
- **Data Preprocessing:** Handles categorical data encoding (Label Encoding for Gender, One-Hot Encoding for Geography) and feature scaling.
- **Deep Learning Model:** Uses a trained Artificial Neural Network (ANN) saved as an `.h5` file.

## Technologies Used

- **Python 3.10–3.13** (TensorFlow 2.21 does not provide installable wheels for Python 3.14 yet)
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

## Deploying on Streamlit Community Cloud (share.streamlit.io)

If dependency installation fails with messages like “no wheels with a matching Python ABI” or “No matching distribution found for tensorflow”, the host is almost certainly using **Python 3.14**. TensorFlow 2.21 is built for **Python 3.10 through 3.13** only, so `pip` cannot install it on 3.14.

**Fix:** When you deploy (or redeploy) the app, open **Advanced settings**, set **Python version** to **3.12** or **3.13**, then save and deploy. You cannot change the Python version after deployment without deleting the app and creating it again with the correct version. Copy your secrets if you use them, then redeploy with the same repository and entrypoint (`app.py`).

Ensure `model.h5` and the `.pkl` files are committed to the repository; the cloud clone must contain them next to `app.py`.

## Project Files

| File | Description |
|------|-------------|
| `app.py` | The main Streamlit web application script. |
| `model.h5` | The trained Artificial Neural Network model. |
| `scaler.pkl` | The saved StandardScaler object used to normalize numerical input features. |
| `label_encoder_gender.pkl` | The saved LabelEncoder object for the `Gender` column. |
| `onehot_encoder_geo.pkl` | The saved OneHotEncoder object for the `Geography` column. |

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
