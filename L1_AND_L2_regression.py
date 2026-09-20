print("Script starting...")
import importlib.util
from sklearn.linear_model import LogisticRegression

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

for c_value in [1, 0.1, 0.01, 0.001]:
    print(f"Fitting L1 sweep, C={c_value}...")
    m_l1 = LogisticRegression(max_iter=2000, l1_ratio=1.0, C=c_value, class_weight='balanced', solver='saga')
    m_l1.fit(X_train_scaled, y_train)
    nonzero = (m_l1.coef_[0] != 0).sum()
    print(f"C={c_value}: L1 = {nonzero}/{X.shape[1]} non-zero coefficients")