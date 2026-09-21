print("Script starting...")
import importlib.util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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
# Balanced model
# ---------------------------------------------------------
print("Fitting balanced model...")
modelbalanced = LogisticRegression(max_iter=1000, class_weight='balanced')
modelbalanced.fit(X_train_scaled, y_train)
print("Balanced model fit done.")

y_predbalanced = modelbalanced.predict(X_test_scaled)
y_probbalanced = modelbalanced.predict_proba(X_test_scaled)[:, 1]

print("Balanced")
print(confusion_matrix(y_test, y_predbalanced))
 
print(f"Accuracy:  {accuracy_score(y_test, y_predbalanced):.2f}")
print(f"Precision: {precision_score(y_test, y_predbalanced):.2f}")
print(f"Recall:    {recall_score(y_test, y_predbalanced):.2f}")
print(f"F1:        {f1_score(y_test, y_predbalanced):.2f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_probbalanced):.2f}")
print("Baseline script finished.")