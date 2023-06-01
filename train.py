import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = pd.read_csv("dataset.csv", header=None, names=["radius", "depth", "bbox_x", "bbox_y"])

# Separate the data into features (X) and target variable (y)
X = data.drop("radius", axis=1)
y = data["radius"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Train the Ridge Regression model and search for optimal hyperparameters
ridge = Ridge()
params_ridge = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
grid_ridge = GridSearchCV(estimator=ridge, param_grid=params_ridge, scoring='r2', cv=5)
grid_ridge.fit(X_train, y_train)
best_ridge = grid_ridge.best_estimator_
print("Best parameters for Ridge Regression:", grid_ridge.best_params_)

# Train the Lasso Regression model and search for optimal hyperparameters
lasso = Lasso()
params_lasso = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
grid_lasso = GridSearchCV(estimator=lasso, param_grid=params_lasso, scoring='r2', cv=5)
grid_lasso.fit(X_train, y_train)
best_lasso = grid_lasso.best_estimator_
print("Best parameters for Lasso Regression:", grid_lasso.best_params_)

# Train the Elastic Net Regression model and search for optimal hyperparameters
elastic = ElasticNet()
params_elastic = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
grid_elastic = GridSearchCV(estimator=elastic, param_grid=params_elastic, scoring='r2', cv=5)
grid_elastic.fit(X_train, y_train)
best_elastic = grid_elastic.best_estimator_
print("Best parameters for Elastic Net Regression:", grid_elastic.best_params_)

# Train the SVM model and search for optimal hyperparameters
svm = SVR()
params_svm = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 1]}
grid_svm = GridSearchCV(estimator=svm, param_grid=params_svm, scoring='r2', cv=5)
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
print("Best parameters for SVM:", grid_svm.best_params_)

# Train the Random Forest model and search for optimal hyperparameters
random_forest = RandomForestRegressor()
params_rf = {'n_estimators': [100, 500, 1000], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(estimator=random_forest, param_grid=params_rf, scoring='r2', cv=5)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("Best parameters for Random Forest:", grid_rf.best_params_)

# Train the models and evaluate their performances
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": best_ridge,
    "Lasso": best_lasso,
    "Elastic Net": best_elastic,
    "SVM": best_svm,
    "Random Forest": best_rf,
    "Voting": VotingRegressor(estimators=[('ridge', best_ridge), ('lasso', best_lasso), ('randomforest', best_rf), ('elastic', best_elastic), ('linear', LinearRegression())])
}

results = {}
best_model = None
best_score = -float('inf')

for model_name, model in models.items():
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    results[model_name] = score

    if score > best_score:
        best_model = model_name
        best_score = score

# Print the scores and the best model
for model_name, score in results.items():
    print(model_name, "R^2 Score:", score)

print("Best model:", best_model)
print("Best score:", best_score)

# Visualize the performances
plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values(), color=(175/255, 173/255, 171/255))
plt.xlabel("Model")
plt.ylabel("R^2 Score")
plt.title("Model Performance Comparison")
plt.ylim(0.2, 1)
plt.show()

# 3D visualization
for model_name, model in models.items():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    predicted_values = model.predict(X_test)
    ax.scatter(X_test['depth'], X_test['bbox_x'] * X_test['bbox_y'], predicted_values, c=predicted_values, cmap='viridis')
    ax.set_xlabel('Depth in m')
    ax.set_ylabel('Bbox area in pixels')
    ax.set_zlabel('Predicted Radius')
    ax.set_title(f'Predicted Radius with Depth and Bbox area - {model_name}')
    plt.show()

import joblib
joblib.dump(best_model, "regression_model.joblib")
