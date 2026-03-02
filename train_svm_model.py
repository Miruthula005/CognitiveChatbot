import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("../Notebooks/dataset_updated.csv")

# Drop unwanted columns
columns_to_drop = ['Age', 'Gender']
df_model = df.drop(columns=columns_to_drop)

# Clean column names
df_model.columns = df_model.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(df_model['Learner'])

# Prepare features
X = df_model.drop('Learner', axis=1)
y = y_encoded

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train SVM
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
print("\n===== SVM Model =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and scaler
model_path = "../saved_model/Learning Style/svm_LS_model.pkl"
scaler_path = "../saved_model/Learning Style/svm_scaler.pkl"

joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)

print("\nModel and scaler saved successfully!")