# Importing required libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Generating data
def generate_data(
    n_samples=100, 
    n_features=20, 
    n_informative=2,
    n_classes=2, 
    n_clusters_per_class=2,
    class_sep=1.0, 
    shuffle=True, 
    random_state=0
):
    return make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative, 
        n_classes=n_classes, 
        n_clusters_per_class=n_clusters_per_class, 
        shuffle=shuffle, 
        random_state=random_state
    )

# Creating and separating the dataset
data_x, data_y = generate_data(n_samples=10000, n_features=4, n_informative=2, n_classes=2, 
                               n_clusters_per_class=2, class_sep=1.0)

X_train, X_test_val, y_train, y_test_val = train_test_split(data_x, data_y, test_size=.3, random_state=0, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=1/3, random_state=0, shuffle=True)

print(y_train.shape[0], y_test.shape[0], y_val.shape[0], type(y_val))

# Saving validation data
np.savez("validation_data.npz", X=X_val, y=y_val)

# Initialize models
svm_model = SVC(kernel='linear')
decision_tree_model = DecisionTreeClassifier(random_state=0)
logistic_regression_model = LogisticRegression(random_state=0, max_iter=1000)

# Perform Cross-Validation and Train Models
def perform_cross_validation(model, X, y, model_name, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\nCross-Validation Results for {model_name}:")
    print(f"Accuracy: {scores.mean():.2f} ± {scores.std():.2f} (over {cv} folds)")
    return scores

# Cross-validation for SVM
perform_cross_validation(svm_model, X_train, y_train, "SVM")

# Train SVM model
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("\nSVM Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_svm, average='weighted'):.2f}")
joblib.dump(svm_model, 'svm_model.pkl')

# Cross-validation for Decision Tree
perform_cross_validation(decision_tree_model, X_train, y_train, "Decision Tree")

# Train Decision Tree model
decision_tree_model.fit(X_train, y_train)
y_pred_tree = decision_tree_model.predict(X_test)
print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_tree, average='weighted'):.2f}")
joblib.dump(decision_tree_model, 'decision_tree_model.pkl')

# Cross-validation for Logistic Regression
perform_cross_validation(logistic_regression_model, X_train, y_train, "Logistic Regression")

# Train Logistic Regression model
logistic_regression_model.fit(X_train, y_train)
y_pred_logistic = logistic_regression_model.predict(X_test)
print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_logistic, average='weighted'):.2f}")
joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl')

print("\nModels saved to respective files.")

