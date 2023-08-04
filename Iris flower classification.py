import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# Example for predicting a new flower
# Replace the values below with the features of the new flower you want to classify
new_flower_features = [5.1, 3.5, 1.4, 0.2]

# Make predictions for the new flower
predicted_class = model.predict([new_flower_features])

print(f"Predicted Class for the new flower: {iris.target_names[predicted_class][0]}")
