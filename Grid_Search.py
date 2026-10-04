print("Script starting...")
import importlib.util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



print("Importing baseline (this re-runs the full baseline script)...")
spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\Test\data_science\E-Commerce_Return_Prediction\Linear_Logistic_Regression_Basecode.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
print("Baseline import finished.")

X_train_scaled = baseline.X_train_scaled
X_test_scaled = baseline.X_test_scaled
X_train = baseline.X_train
X_test = baseline.X_test
y_train = baseline.y_train
y_test = baseline.y_test


# ---------------------------------------------------------
# GRID SEARCH — Logistic Regression
# ---------------------------------------------------------
print("Running grid search (Logistic Regression, up to 50 fits)...")
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced']}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, scoring='f1', cv=5, n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)
print("Logistic Regression grid search done.")
 
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
 
print("\nLogistic Regression — Best params:", grid_lr.best_params_)
print("Best CV F1 score:", grid_lr.best_score_)
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
 
# ---------------------------------------------------------
# GRID SEARCH — CART (Decision Tree)
# ---------------------------------------------------------
print("\nRunning grid search (CART / Decision Tree)...")
param_grid_cart = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': [None, 'balanced']
}
grid_cart = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_cart, scoring='f1', cv=5, n_jobs=-1)
grid_cart.fit(X_train, y_train)
print("CART grid search done.")
 
best_cart = grid_cart.best_estimator_
y_pred_cart = best_cart.predict(X_test)
 
print("\nCART — Best params:", grid_cart.best_params_)
print("Best CV F1 score:", grid_cart.best_score_)
print(confusion_matrix(y_test, y_pred_cart))
print(classification_report(y_test, y_pred_cart))
 
# ---------------------------------------------------------
# GRID SEARCH — Random Forest
# ---------------------------------------------------------
print("\nRunning grid search (Random Forest, this may take a while)...")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': [None, 'balanced']
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid_rf, scoring='f1', cv=5, n_jobs=1)
grid_rf.fit(X_train, y_train)
print("Random Forest grid search done.")
 
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
 
print("\nRandom Forest — Best params:", grid_rf.best_params_)
print("Best CV F1 score:", grid_rf.best_score_)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
 