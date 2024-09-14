# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Step 2: Load the dataset
url = 'https://drive.google.com/uc?id=1JOqcR8X0GcFQAivwMzP2cGrTyhZ-4_j2'
df = pd.read_csv(url)

# Step 3: Data exploration and preprocessing
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Convert 'TotalCharges' to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Drop 'customerID' as it is not useful for modeling
df.drop('customerID', axis=1, inplace=True)

# Encode categorical variables
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df = pd.get_dummies(df)

# Step 4: Exploratory Data Analysis (EDA)
# Churn distribution
plt.figure(figsize=(6,6))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Tenure vs. Churn
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack')
plt.title('Tenure vs. Churn')
plt.show()

# Monthly Charges vs. Churn
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack')
plt.title('Monthly Charges vs. Churn')
plt.show()

# Step 5: Data Preparation
# Split the data into features and target variable
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model Development
# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

# Train models and evaluate
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Results for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('-' * 60)

# Step 7: Hyperparameter tuning for the best model (example: RandomForest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Evaluate the best model on test data
y_pred_best_rf = best_rf.predict(X_test)
print("Best Random Forest Model Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("Precision:", precision_score(y_test, y_pred_best_rf))
print("Recall:", recall_score(y_test, y_pred_best_rf))
print("F1 Score:", f1_score(y_test, y_pred_best_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))

# Step 8: Feature Importance (Random Forest example)
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()

# Step 9: Model Deployment
# Save the model (example)
import joblib
joblib.dump(best_rf, 'best_rf_model.pkl')

# Final Report
print("Model training complete. Best Random Forest model saved as 'best_rf_model.pkl'.")

<!---
Devangkedardadhich/Devangkedardadhich is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
