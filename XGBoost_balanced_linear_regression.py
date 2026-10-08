print("Script starting...")
import importlib.util
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("Importing baseline (this re-runs the full baseline script)...")
spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\Test\data_science\E-Commerce_Return_Prediction\Linear_Logistic_Regression_Basecode.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
print("Baseline import finished.")

X_train = baseline.X_train
X_test = baseline.X_test
y_train = baseline.y_train
y_test = baseline.y_test

# ---------------------------------------------------------
# XGBoost
# ---------------------------------------------------------
# XGBoost has no class_weight param; scale_pos_weight (neg/pos ratio) does the same job
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

print("Fitting XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
print("XGBoost fit done.")

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost")
print(confusion_matrix(y_test, y_pred_xgb))

print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred_xgb):.2f}")
print(f"F1:        {f1_score(y_test, y_pred_xgb):.2f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_xgb):.2f}")
print("XGBoost script finished.")