print("Script starting...")
import importlib.util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

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
X = baseline.X

print("Fitting L1...")

c_value = 0.1
model_l1 = LogisticRegression(max_iter=1000, l1_ratio=1.0, C=c_value, class_weight='balanced', solver='saga')
model_l1.fit(X_train_scaled, y_train)
print("L1 done.")

y_pred_l1 = model_l1.predict(X_test_scaled)
y_prob_l1 = model_l1.predict_proba(X_test_scaled)[:, 1]
nonzero_l1 = (model_l1.coef_[0] != 0).sum()

print(f"C={c_value}: L1 = {nonzero_l1}/{X.shape[1]} non-zero coefficients")
print(f"L1 Precision: {precision_score(y_test, y_pred_l1):.2f}")
print(f"L1 Recall:    {recall_score(y_test, y_pred_l1):.2f}")
print(f"L1 F1:        {f1_score(y_test, y_pred_l1):.2f}")
print(f"L1 ROC-AUC:   {roc_auc_score(y_test, y_prob_l1):.2f}")
print("L1 non-zero coefficients:", nonzero_l1, "/", len(model_l1.coef_[0]))

print("\nFitting L2...")

c_value = 0.001
model_l2 = LogisticRegression(max_iter=1000, l1_ratio=0.0, C=c_value, class_weight='balanced', solver='saga')
model_l2.fit(X_train_scaled, y_train)
print("L2 done.")

y_pred_l2 = model_l2.predict(X_test_scaled)
y_prob_l2 = model_l2.predict_proba(X_test_scaled)[:, 1]
nonzero_l2 = (model_l2.coef_[0] != 0).sum()

print(f"C={c_value}: L2 = {nonzero_l2}/{X.shape[1]} non-zero coefficients")
print(f"L2 Precision: {precision_score(y_test, y_pred_l2):.2f}")
print(f"L2 Recall:    {recall_score(y_test, y_pred_l2):.2f}")
print(f"L2 F1:        {f1_score(y_test, y_pred_l2):.2f}")
print(f"L2 ROC-AUC:   {roc_auc_score(y_test, y_prob_l2):.2f}")
print("L2 non-zero coefficients:", nonzero_l2, "/", len(model_l2.coef_[0]))