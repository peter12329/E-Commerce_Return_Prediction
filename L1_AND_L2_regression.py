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
model_l1 = LogisticRegression(max_iter=1000, l1_ratio=1.0, class_weight='balanced', solver='saga')
model_l1.fit(X_train_scaled, y_train)
print("L1 done.")

print("Fitting L2...")
model_l2 = LogisticRegression(max_iter=1000, l1_ratio=0.0, class_weight='balanced', solver='saga')
model_l2.fit(X_train_scaled, y_train)
print("L2 done.")

print("L1 non-zero coefficients:", (model_l1.coef_[0] != 0).sum(), "/", len(model_l1.coef_[0]))
print("L2 non-zero coefficients:", (model_l2.coef_[0] != 0).sum(), "/", len(model_l2.coef_[0]))

c_value = 0.1
nonzero = (model_l1.coef_[0] != 0).sum()

y_pred_c = model_l1.predict(X_test_scaled)
y_prob_c = model_l1.predict_proba(X_test_scaled)[:, 1]

print(f"C={c_value}: L1 = {nonzero}/{X.shape[1]} non-zero coefficients")
print(f"  Precision: {precision_score(y_test, y_pred_c):.2f}")
print(f"  Recall:    {recall_score(y_test, y_pred_c):.2f}")
print(f"  F1:        {f1_score(y_test, y_pred_c):.2f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob_c):.2f}")