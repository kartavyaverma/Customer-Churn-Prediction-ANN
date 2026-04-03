# Customer Churn Prediction using ANN

This project is a machine learning web application built with Streamlit that predicts whether a bank customer will churn (leave the bank) or not based on various customer demographics and account information.

The prediction model uses an Artificial Neural Network (ANN) built with TensorFlow/Keras.

---

## Features

* **Interactive Web UI:** A user-friendly Streamlit interface to input customer details.
* **Real-time Prediction:** Instantly predicts the probability of a customer churning.
* **Data Preprocessing:** Handles categorical data encoding (Label Encoding for Gender, One-Hot Encoding for Geography) and feature scaling.
* **Deep Learning Model:** Uses a trained Artificial Neural Network (ANN) saved as an `.h5` file.

---

## Technologies Used

* **Python 3.10–3.13**
* **Streamlit:** For the web application frontend and backend.
* **TensorFlow & Keras:** For the Artificial Neural Network model.
* **Scikit-Learn:** For data preprocessing (StandardScaler, LabelEncoder, OneHotEncoder).
* **Pandas & NumPy:** For data manipulation and arrays.

---

## Installation and Setup

Clone the repository or navigate to the project directory:

```bash
cd "Customer-Churn Prediction"
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**

```bash
.\venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Deployment (Streamlit Cloud)

If you get TensorFlow errors like:

* *No matching distribution found*
* *No wheels available*

👉 Set Python version to:

```
3.12 or 3.13
```

Ensure these files are present in repo:

* `model.h5`
* `.pkl` files

---

## Project Files

| File                       | Description                        |
| -------------------------- | ---------------------------------- |
| `app.py`                   | Main Streamlit application         |
| `model.h5`                 | Trained ANN model                  |
| `scaler.pkl`               | StandardScaler for feature scaling |
| `label_encoder_gender.pkl` | LabelEncoder for Gender            |
| `onehot_encoder_geo.pkl`   | OneHotEncoder for Geography        |
| `experiments.ipynb`        | Model training notebook            |
| `prediction.ipynb`         | Testing predictions                |
| `requirements.txt`         | Dependencies                       |

---

## Input Parameters

| Parameter          | Description                      |
| ------------------ | -------------------------------- |
| Geography          | Country (France, Spain, Germany) |
| Gender             | Male or Female                   |
| Age                | Customer age                     |
| Balance            | Account balance                  |
| Credit Score       | Credit score                     |
| Estimated Salary   | Annual salary                    |
| Tenure             | Years with bank                  |
| Number of Products | Bank products used               |
| Has Credit Card    | 1 = Yes, 0 = No                  |
| Is Active Member   | 1 = Yes, 0 = No                  |

---

## Prediction Logic

* Model outputs probability (0 → 1)
* **> 0.5 → Churn**
* **≤ 0.5 → Not churn**

---

## Notes

* Logs and unnecessary files are excluded using `.gitignore`
* Model and preprocessing artifacts are saved for reproducibility

---
