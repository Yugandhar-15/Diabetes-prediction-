
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
df = pd.read_csv(url, header=None)


df.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
              "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
print(df.head())
print(df.tail())



# 🔹 Handle missing values (important before standardization)
cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Data Standardization using :contentReference[oaicite:0]{index=0}
scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using same scaler
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


# 🔹 Prediction Function (with standardization applied)
def predict_diabetes(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Apply SAME scaler
    std_data = scaler.transform(input_df)
    
    prediction = model.predict(std_data)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"


# Test input
sample_input = (6,148,72,35,0,33.6,0.627,50)
print("Prediction:", predict_diabetes(sample_input))