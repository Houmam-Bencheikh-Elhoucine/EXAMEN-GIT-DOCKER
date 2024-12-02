from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import joblib

# loading th model
svm = joblib.load("svm_model.pkl")
print("Model loaded successfully!")

# loading validation data
loaded_data = np.load("validation_data.npz")
X_val = loaded_data["X"]
y_val = loaded_data["y"]

print("Validation data loaded successfully!")

# running a prediction
y_pred = svm.predict(X_val)
print("Predictions successful")

# Evaluating data
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"""
Model's performance:
Accuracy: {accuracy:.2f}
Recall: {recall: .2f}
Precision: {precision: .2f}
F1: {f1:.2f}
"""
)