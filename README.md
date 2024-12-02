# EXAMEn-GIT-DOCKER
## Introduction

train_classifier.py script provides a workflow for training and optimizing machine learning models for binary classification tasks. The dataset is synthetically generated using make_classification from sklearn, and hyperparameters for the models are optimized using GridSearchCV.

### Three models are optimized:

    Support Vector Machine (SVM)
    Decision Tree
    Logistic Regression

Each model is evaluated using Accuracy and F1 Score metrics.

### Requirements
To run this script, you will need the following Python libraries:

    numpy
    scikit-learn
    joblib

You can install them using pip:

pip install numpy scikit-learn joblib

### The script will:
        Generate synthetic data.
        Split it into training, validation, and test sets.
        Optimize the three machine learning models using GridSearchCV.
        Evaluate the models on the test set.
        Save the best models as .pkl files for future use.


### predict_classifer
loads pre-trained machine learning models (SVM, Decision Tree, and Logistic Regression), loads validation data, and evaluates the models' performance using accuracy, recall, precision, and F1 score metrics.

### to run this code you need :
  -download docker image form docker hub : docker pull boualem775/examen-git-docker-app
  -to run the image :docker run boualem775/examen-git-docker-app
