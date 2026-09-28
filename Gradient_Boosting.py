print("Script starting...")
import importlib.util
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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


print("Fitting Gradient Boosting...")
gb_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.1,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
gb_model.fit(X_train, y_train)
print("Gradient Boosting fit done.")

y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

print("\nGradient Boosting")
print(confusion_matrix(y_test, y_pred_gb))
print(pd.Series(y_pred_gb).value_counts())

print(f"Accuracy:  {accuracy_score(y_test, y_pred_gb):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_gb):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred_gb):.2f}")
print(f"F1:        {f1_score(y_test, y_pred_gb):.2f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_gb):.2f}")