print("Script starting...")
import importlib.util
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("Importing baseline (this re-runs the full baseline script)...")
spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\Linear_Logistic_Regression_Basecode.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
print("Baseline import finished.")

X_train = baseline.X_train
X_test = baseline.X_test
y_train = baseline.y_train
y_test = baseline.y_test

# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------
print("Fitting Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced_subsample",  # re-balances classes inside each tree's bootstrap sample
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)
print("Random Forest fit done.")

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest")
print(confusion_matrix(y_test, y_pred_rf))
print(pd.Series(y_pred_rf).value_counts())

print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.2f}")
print(f"F1:        {f1_score(y_test, y_pred_rf):.2f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.2f}")

# ---------------------------------------------------------
# Top 10 most important features
# ---------------------------------------------------------
feature_names = getattr(X_train, "columns", None)
if feature_names is None:
    feature_names = getattr(getattr(baseline, "X", None), "columns", None)

if feature_names is not None:
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    print("\nTop 10 feature importances")
    print(importances.sort_values(ascending=False).head(10))
else:
    print("\nFeature names not available (X_train is a plain array); skipping importances.")