from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import joblib

# Load models
svm = joblib.load("svm_model.pkl")
decision_tree = joblib.load("decision_tree_model.pkl")
logistic_regression = joblib.load("logistic_regression_model.pkl")

print("Models loaded successfully!")

# Load validation data
loaded_data = np.load("validation_data.npz")
X_val = loaded_data["X"]
y_val = loaded_data["y"]

print("Validation data loaded successfully!")

# Define a function to evaluate a model
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"""
    {model_name} Performance:
    Accuracy: {accuracy:.2f}
    Recall: {recall:.2f}
    Precision: {precision:.2f}
    F1 Score: {f1:.2f}
    """)
    return 

# Evaluate SVM
evaluate_model(svm, X_val, y_val, "SVM")

# Evaluate Decision Tree
evaluate_model(decision_tree, X_val, y_val, "Decision Tree")

# Evaluate Logistic Regression
evaluate_model(logistic_regression, X_val, y_val, "Logistic Regression")

