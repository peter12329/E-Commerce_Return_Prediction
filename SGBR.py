import importlib.util

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

print("Script starting...")

print("Importing baseline (this re-runs the full baseline script)...")

spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\Test\data_science\E-Commerce_Return_Prediction\Linear_Logistic_Regression_Basecode.py"
)

baseline = importlib.util.module_from_spec(spec)

spec.loader.exec_module(baseline)

print("Baseline import finished.")

X_train = baseline.X_train
X_test = baseline.X_test

y_train = baseline.y_train
y_test = baseline.y_test


# ---------------------------------------------------------
# Stochastic Gradient Boosting Regressor
# ---------------------------------------------------------

print("Fitting SGBR...")

sgbr_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

sgbr_model.fit(X_train, y_train)

print("SGBR fit done.")

y_pred_sgbr = sgbr_model.predict(X_test)


# ---------------------------------------------------------
# SGBR Evaluation
# ---------------------------------------------------------

print("\nSGBR")

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_sgbr):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_sgbr):.4f}")
print(f"R²: {r2_score(y_test, y_pred_sgbr):.4f}")


# ---------------------------------------------------------
# Cross Validation
# ---------------------------------------------------------

cv_scores = cross_val_score(
    sgbr_model,
    X_train,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

cv_mse = -cv_scores

print(f"Cross-validated MSE scores: {cv_mse}")
print(f"Mean CV MSE: {cv_mse.mean():.4f}")


# ---------------------------------------------------------
# Actual vs Predicted
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))

plt.scatter(y_test, y_pred_sgbr)

plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.title("SGBR Actual vs Predicted")

plt.savefig(
    "SGBR_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Feature Importances
# ---------------------------------------------------------

importances = sgbr_model.feature_importances_

features = X_train.columns

feat_imp = pd.Series(
    importances,
    index=features
).sort_values(ascending=False)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=feat_imp.values,
    y=feat_imp.index
)

plt.title("SGBR Feature Importances")

plt.savefig(
    "SGBR_feature_importances.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Plots saved successfully.")