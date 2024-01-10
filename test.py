from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Load a simple dataset
X, y = load_iris(return_X_y=True)

# Create a simple decision tree classifier
model = DecisionTreeClassifier()

# Define a parameter distribution
param_dist = {"max_depth": [3, None],
              "max_features": np.arange(1, 11),
              "min_samples_leaf": np.arange(1, 11)}

# Set up RandomizedSearchCV
rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, verbose=3, n_jobs=-1)

# Fit the model (this will generate verbose output)
rs.fit(X, y)
