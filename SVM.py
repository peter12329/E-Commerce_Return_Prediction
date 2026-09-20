print("Script starting...")
import importlib.util
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("Importing baseline (this re-runs the full baseline script)...")
spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\linear logistic regression implementation.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
print("Baseline import finished.")

X_train_scaled = baseline.X_train_scaled
X_test_scaled = baseline.X_test_scaled
y_train = baseline.y_train
y_test = baseline.y_test

# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------
print("Setting up SVM...")
svm_model = LinearSVC(max_iter=5000, class_weight='balanced', random_state=42)
svm_calibrated = CalibratedClassifierCV(svm_model, cv=3)

print("Fitting SVM (this trains internally 3x for calibration, may take a while)...")
svm_calibrated.fit(X_train_scaled, y_train)
print("SVM fit done.")

y_pred_svm = svm_calibrated.predict(X_test_scaled)
y_prob_svm = svm_calibrated.predict_proba(X_test_scaled)[:, 1]

print("\nSVM")
print(confusion_matrix(y_test, y_pred_svm))
print(pd.Series(y_pred_svm).value_counts())

print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_svm):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_svm):.2f}")
print(f"F1: {f1_score(y_test, y_pred_svm):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_svm):.2f}")