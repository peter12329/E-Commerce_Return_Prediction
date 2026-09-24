print("Script starting...")
import importlib.util
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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

print("Fitting Bagging (100 trees, OOB on)...")
model_bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(class_weight='balanced', random_state=42),
    n_estimators=100,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
model_bag.fit(X_train_scaled, y_train)
print("Bagging done.")

print(f"OOB score (accuracy): {model_bag.oob_score_:.4f}")
print(f"OOB error:            {1 - model_bag.oob_score_:.4f}")

y_pred = model_bag.predict(X_test_scaled)
y_prob = model_bag.predict_proba(X_test_scaled)[:, 1]
print("Bagging test results:")
print(f"  Precision: {precision_score(y_test, y_pred):.2f}")
print(f"  Recall:    {recall_score(y_test, y_pred):.2f}")
print(f"  F1:        {f1_score(y_test, y_pred):.2f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob):.2f}")

for n_trees in [10, 50, 100, 200]:
    print(f"Fitting Bagging sweep, n_estimators={n_trees}...")
    m_bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(class_weight='balanced', random_state=42),
        n_estimators=n_trees,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    m_bag.fit(X_train_scaled, y_train)

    y_pred_n = m_bag.predict(X_test_scaled)
    y_prob_n = m_bag.predict_proba(X_test_scaled)[:, 1]

    print(f"n_estimators={n_trees}: OOB score = {m_bag.oob_score_:.4f} (OOB error = {1 - m_bag.oob_score_:.4f})")
    print(f"  Precision: {precision_score(y_test, y_pred_n):.2f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_n):.2f}")
    print(f"  F1:        {f1_score(y_test, y_pred_n):.2f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob_n):.2f}")