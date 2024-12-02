# Importing required libraries
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# generating data
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

# creating and seperating dataset

data_x, data_y = generate_data(n_samples=10000, n_features=4, n_informative=2, n_classes=2, 
                                                   n_clusters_per_class=2, 
                                                   class_sep=1.0)

X_train, X_test_val, y_train, y_test_val = train_test_split(data_x, data_y, test_size=.3, random_state=0, shuffle=True)

X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=1/3, random_state=0, shuffle=True)


print(y_train.shape[0], y_test.shape[0], y_val.shape[0], type(y_val))

# saving validation data
np.savez("validation_data.npz", X=X_val, y=y_val)