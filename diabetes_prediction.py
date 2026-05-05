import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")

print(df.head())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
print(X.head())
print(y.tail())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_diabetes(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    std_data = scaler.transform(input_df)
    prediction = model.predict(std_data)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

sample_input = (6,148,72,35,0,33.6,0.627,50)
print("Prediction:", predict_diabetes(sample_input))
