print("Script starting...")
import importlib.util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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
# Balanced model
# ---------------------------------------------------------
print("Fitting balanced model...")
modelbalanced = LogisticRegression(max_iter=1000, class_weight='balanced')
modelbalanced.fit(X_train_scaled, y_train)
print("Balanced model fit done.")

y_predbalanced = modelbalanced.predict(X_test_scaled)
y_probbalanced = modelbalanced.predict_proba(X_test_scaled)[:, 1]

print("\nBalanced model confusion matrix:")
print(confusion_matrix(y_test, y_predbalanced))