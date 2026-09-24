print("Script starting...")
import importlib.util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

print("Importing baseline (this re-runs the full baseline script)...")
spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\Linear_Logistic_Regression_Basecode.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
print("Baseline import finished.")

X_train_scaled = baseline.X_train_scaled
X_test_scaled = baseline.X_test_scaled
y_train = baseline.y_train
y_test = baseline.y_test

# ---------------------------------------------------------
# GRID SEARCH
# ---------------------------------------------------------
print("Running grid search (this fits up to 50 models, may take a while)...")
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced']}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, scoring='f1', cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print("Grid search done.")

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nBest params:", grid.best_params_)
print("Best CV F1 score:", grid.best_score_)

print("\nGrid search best model confusion matrix:")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))