import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the data from CSV file
data = pd.read_csv("CREDITSCORE.csv")

# Perform one-hot encoding for the 'Credit_Mix' column
data = pd.get_dummies(data, columns=["Credit_Mix"])

# Separate features and target variable
X = data[["Annual_Income", "Monthly_Inhand_Salary",
          "Num_Bank_Accounts", "Num_Credit_Card",
          "Interest_Rate", "Num_of_Loan",
          "Delay_from_due_date", "Num_of_Delayed_Payment",
          "Outstanding_Debt",
          "Credit_History_Age", "Monthly_Balance",
          "Credit_Mix_Bad", "Credit_Mix_Standard", "Credit_Mix_Good"]]
y = data["Credit_Score"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create and train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict Credit Score for a new entry
print("Credit Score Prediction:")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
credit_mix = [0, 0, 0]  # Default values for all categories
i = int(input("Credit Mix (Bad: 0, Standard: 1, Good: 2): "))
credit_mix[i] = 1  # Set the corresponding category to 1 based on user input

j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

# Make prediction for the new entry
features = np.array([[a, b, c, d, e, f, g, h, j, k, l] + credit_mix])
predicted_score = model.predict(features)
print("Predicted Credit Score =", predicted_score[0])
