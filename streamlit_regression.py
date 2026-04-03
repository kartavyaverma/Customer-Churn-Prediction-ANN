import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import pickle

# Load the trained regression model
model = tf.keras.models.load_model('regression_model.h5', compile=False)

# Load the one-hot encoder saved from salary notebook
with open('onehot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

# Recreate preprocessing objects used during training
df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

label_encoder_gender = LabelEncoder()
df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])

y = df['EstimatedSalary']
X = df.drop(columns=['EstimatedSalary'])

geo_encoded_train = one_hot_encoder.transform(X[['Geography']]).toarray()
geo_encoded_train_df = pd.DataFrame(
    geo_encoded_train,
    columns=one_hot_encoder.get_feature_names_out(['Geography']),
    index=X.index,
)

X = pd.concat([X.drop(columns=['Geography']), geo_encoded_train_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)

# Define the Streamlit app
st.title("Customer Salary Regression Prediction")

# Define input fields for user to enter customer data
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode Geography (DataFrame preserves column name for sklearn)
geo_encoded = one_hot_encoder.transform(
    pd.DataFrame({'Geography': [geography]})
).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_data, verbose=0)
predicted_salary = float(prediction[0][0])

# Display the prediction result
st.write(f"Predicted Estimated Salary: {predicted_salary:,.2f}")
