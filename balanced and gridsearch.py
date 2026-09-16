# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.utils.class_weight import compute_class_weight

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# Variables from the baseline
# Only pulling what this script actually uses downstream
# (y_pred / y_prob / model, for the commented-out comparison
# table). X_train_scaled, X_test_scaled, y_train, y_test are
# NOT imported — this script builds its own split below, and
# importing-then-overwriting them was wasted work that also
# risked masking a preprocessing mismatch between the two files.
# ---------------------------------------------------------
import importlib.util

spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\linear logistic regression implementation.py"
)

baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

y_pred = baseline.y_pred
y_prob = baseline.y_prob
model = baseline.model

# ---------------------------------------------------------
# Same preprocessing as the base script (kept identical so
# results are directly comparable)
# ---------------------------------------------------------
path = r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\datasets\ecommerce_sales_customer_analytics_150k.csv"
df = pd.read_csv(path)

df['is_returned'] = df['return_status'].notna().astype(int)

leak_or_useless = [
    'order_id', 'customer_id', 'customer_name',
    'order_date', 'order_time',

    'return_status', 'return_reason',
    'customer_review', 'review_sentiment',
    'customer_rating',

    'campaign_name', 'coupon_code',

    'customer_postal_code', 'customer_city',

    'order_status', 'payment_status',

    'loyalty_points_earned',
    'discount_amount',
    'loyalty_points_redeemed',
    'gross_sales',

    'delivery_status',
    'delivery_days',
    'estimated_delivery_days',

    'net_sales',
    'profit',
    'profit_margin_percentage',
]
df = df.drop(columns=[c for c in leak_or_useless if c in df.columns])

suspect_cols = ['loyalty_points_earned', 'discount_amount', 'loyalty_points_redeemed', 'gross_sales']

#for col in suspect_cols:
#    print(f"\n{col}:")
#    print(df.groupby('is_returned')[col].describe())

#df['delivery_days'] = df['delivery_days'].fillna(df['delivery_days'].median())
#df['estimated_delivery_days'] = df['estimated_delivery_days'].fillna(df['estimated_delivery_days'].median())


# ---------------------------------------------------------
# Data split + encoding
#
# Instead of pd.get_dummies() on the whole feature set followed
# by StandardScaler on everything (which also standardizes the
# 0/1 dummy columns and distorts their coefficients), we split
# columns into numeric vs categorical up front and scale only
# the numeric ones. Categorical columns are one-hot encoded but
# left as clean 0/1 indicators in both versions below.
#
# Two encoded versions are produced:
#   - X_train_scaled / X_test_scaled -> numeric cols standardized,
#     used by Logistic Regression and SVM (coefficients stay
#     directly comparable/interpretable for the leakage check).
#   - X_train / X_test -> numeric cols left raw, used by the
#     Decision Tree (doesn't need scaling).
# ---------------------------------------------------------
X_raw = df.drop(columns=['is_returned'])
y = df['is_returned']

numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print("Class weights:", dict(zip(np.unique(y_train), weights)))

# handle_unknown='ignore' protects transform() if the test split
# ever contains a category the train split didn't see.
# sparse_output requires sklearn >= 1.2; use sparse=False on older versions.
scaling_ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
    ]
)
X_train_scaled = scaling_ct.fit_transform(X_train_raw)
X_test_scaled = scaling_ct.transform(X_test_raw)

encode_ct = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
    ]
)
X_train = encode_ct.fit_transform(X_train_raw)
X_test = encode_ct.transform(X_test_raw)

# Column order matches X_train_scaled / X_test_scaled — used later
# for the coefficient leakage check.
feature_names = scaling_ct.get_feature_names_out()

# ---------------------------------------------------------
# GRID SEARCH
# ---------------------------------------------------------
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced']}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, scoring='f1', cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nBest params:", grid.best_params_)
print("Best CV F1 score:", grid.best_score_)

print("\nGrid search best model confusion matrix:")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# ---------------------------------------------------------
# Balanced model
# ---------------------------------------------------------
modelbalanced = LogisticRegression(max_iter=1000, class_weight='balanced')
modelbalanced.fit(X_train_scaled, y_train)

y_predbalanced = modelbalanced.predict(X_test_scaled)
y_probbalanced = modelbalanced.predict_proba(X_test_scaled)[:, 1]

print("\nBalanced model confusion matrix:")
print(confusion_matrix(y_test, y_predbalanced))


# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------

svm_model = LinearSVC(max_iter=5000, class_weight='balanced', random_state=42)
svm_calibrated = CalibratedClassifierCV(svm_model, cv=3)
svm_calibrated.fit(X_train_scaled, y_train)

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


# ---------------------------------------------------------
# DECISION TREE
# ---------------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 5},
    random_state=42
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

print("\nDecision Tree")
print(confusion_matrix(y_test, y_pred_tree))
print(pd.Series(y_pred_tree).value_counts())

print(f"Accuracy:  {accuracy_score(y_test, y_pred_tree):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_tree):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred_tree):.2f}")
print(f"F1:        {f1_score(y_test, y_pred_tree):.2f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_tree):.2f}")


# ---------------------------------------------------------
# ROC
# ---------------------------------------------------------

#RocCurveDisplay.from_estimator(modelbalanced, X_test_scaled, y_test)
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
#plt.title('ROC Curve — Return Prediction (Balanced)')
#plt.legend()
#plt.show()

# importance_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': tree_model.feature_importances_
# }).sort_values('importance', ascending=False).head(10)
# print(importance_df)

# ---------------------------------------------------------
# Comparison table
# ---------------------------------------------------------
# print("Number of records:", df.shape[0])
# print("Number of model features:", X_train_scaled.shape[1])
# print("Target variable: is_returned")
# comparison = pd.DataFrame({
#     'Metric': ['Accuracy', 'Precision (Returned)', 'Recall (Returned)', 'F1 (Returned)', 'ROC-AUC'],
#     'Baseline': [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_prob)],
#     'Balanced': [accuracy_score(y_test, y_predbalanced), precision_score(y_test, y_predbalanced), recall_score(y_test, y_predbalanced), f1_score(y_test, y_predbalanced), roc_auc_score(y_test, y_probbalanced)],
#     'Grid Search': [accuracy_score(y_test, y_pred_best), precision_score(y_test, y_pred_best), recall_score(y_test, y_pred_best), f1_score(y_test, y_pred_best), roc_auc_score(y_test, y_prob_best)]
# })
# print(comparison.round(2))


# ---------------------------------------------------------
# Leakage sanity check — are predicted probabilities for
# actual returns suspiciously clustered near 1.0?
# ---------------------------------------------------------
returned_probs = y_prob_best[y_test == 1]
not_returned_probs = y_prob_best[y_test == 0]

print("\nReturned orders — predicted probability stats:")
print(pd.Series(returned_probs).describe())

print("\nNot Returned orders — predicted probability stats:")
print(pd.Series(not_returned_probs).describe())

# ---------------------------------------------------------
# Top coefficients — look for one feature dominating the rest
# (a sign of leakage rather than genuine signal)
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': best_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 feature coefficients (grid search best model):")
print(coef_df.head(10))