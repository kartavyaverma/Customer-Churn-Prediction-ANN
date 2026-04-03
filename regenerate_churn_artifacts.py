"""
Regenerate label_encoder_gender.pkl, onehot_encoder_geo.pkl, scaler.pkl
from Churn_Modelling.csv using the churn training pipeline (matches experiments.ipynb).
Run from project root: python regenerate_churn_artifacts.py
"""
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df = pd.read_csv("Churn_Modelling.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

label_encoder_gender = LabelEncoder()
df["Gender"] = label_encoder_gender.fit_transform(df["Gender"])

onehot_encoder_geo = OneHotEncoder()
geo_encoded = onehot_encoder_geo.fit_transform(df[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
    index=df.index,
)
df = pd.concat([df.drop(columns=["Geography"]), geo_encoded_df], axis=1)

X = df.drop(columns=["Exited"])
y = df["Exited"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)

with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(label_encoder_gender, f)
with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(onehot_encoder_geo, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved label_encoder_gender.pkl, onehot_encoder_geo.pkl, scaler.pkl")
print("Scaler features:", list(scaler.feature_names_in_))
