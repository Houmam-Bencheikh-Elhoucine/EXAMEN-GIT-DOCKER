# Importing required libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Hyperparameter Grids
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

decision_tree_param_grid = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

logistic_regression_param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}

# Optimizing Models with Grid Search
def optimize_model(model, param_grid, X, y, model_name, cv=3):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print(f"\nBest Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy for {model_name}: {grid_search.best_score_:.2f}")
    return grid_search.best_estimator_

# Optimize SVM
optimized_svm = optimize_model(SVC(), svm_param_grid, X_train, y_train, "SVM")

# Evaluate SVM
y_pred_svm = optimized_svm.predict(X_test)
print("\nSVM Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_svm, average='weighted'):.2f}")
joblib.dump(optimized_svm, 'optimized_svm_model.pkl')

# Optimize Decision Tree
optimized_tree = optimize_model(DecisionTreeClassifier(random_state=0), decision_tree_param_grid, X_train, y_train, "Decision Tree")

# Evaluate Decision Tree
y_pred_tree = optimized_tree.predict(X_test)
print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_tree, average='weighted'):.2f}")
joblib.dump(optimized_tree, 'optimized_decision_tree_model.pkl')

# Optimize Logistic Regression
optimized_logistic = optimize_model(LogisticRegression(max_iter=1000), logistic_regression_param_grid, X_train, y_train, "Logistic Regression")

# Evaluate Logistic Regression
y_pred_logistic = optimized_logistic.predict(X_test)
print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_logistic, average='weighted'):.2f}")
joblib.dump(optimized_logistic, 'optimized_logistic_regression_model.pkl')

print("\nOptimized models saved to respective files.")

