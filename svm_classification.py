
# Importing libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# Loading dataset
dataset = pd.read_csv("data/logit_classification.csv")

# Separating independent and dependent variables
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

# Creating a pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])

# Training the SVM model
pipeline.fit(X_train, y_train)

# Making predictions
y_pred = pipeline.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Training and Testing Accuracy
train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Cross Validation
accuracies = cross_val_score(
    estimator=pipeline,
    X=X_train,
    y=y_train,
    cv=10
)

print("\nCross Validation Results:")
print("Mean Accuracy: {:.2f}%".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f}%".format(accuracies.std() * 100))

# Hyperparameter Tuning using GridSearchCV
parameters = [
    {
        "classifier__C": [1, 10, 100, 1000],
        "classifier__kernel": ["linear"]
    },
    {
        "classifier__C": [1, 10, 100, 1000],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9
        ]
    }
]

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameters,
    scoring="accuracy",
    cv=10,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best Parameters
print("\nBest Cross Validation Accuracy: {:.2f}%".format(
    grid_search.best_score_ * 100
))

print("\nBest Parameters:")
print(grid_search.best_params_)

# Evaluating optimized model
best_model = grid_search.best_estimator_

best_predictions = best_model.predict(X_test)

print("\nOptimized Model Test Accuracy:",
      accuracy_score(y_test, best_predictions))

print("\nOptimized Model Classification Report:")
print(classification_report(y_test, best_predictions))


import os
import pickle

os.makedirs("models", exist_ok=True)

with open("models/svm_model.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("SVM model saved successfully!")

