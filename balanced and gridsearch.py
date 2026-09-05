import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------
# Same preprocessing as the base script (kept identical so
# results are directly comparable)
# ---------------------------------------------------------
path = r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\datasets\ecommerce_sales_customer_analytics_150k.csv"
df = pd.read_csv(path)

df['is_returned'] = df['return_status'].notna().astype(int)

leak_or_useless = [
    'order_id', 'customer_id', 'customer_name', 'order_date', 'order_time',
    'return_status', 'return_reason', 'customer_review', 'review_sentiment',
    'customer_rating', 'campaign_name', 'coupon_code',
    'customer_postal_code', 'customer_city',
    'order_status', 'payment_status', 'loyalty_points_earned', 'discount_amount', 'loyalty_points_redeemed', 'gross_sales', 'delivery_status']
df = df.drop(columns=leak_or_useless)

suspect_cols = ['loyalty_points_earned', 'discount_amount', 'loyalty_points_redeemed', 'gross_sales']

#for col in suspect_cols:
#    print(f"\n{col}:")
#    print(df.groupby('is_returned')[col].describe())

df['delivery_days'] = df['delivery_days'].fillna(df['delivery_days'].median())
df['estimated_delivery_days'] = df['estimated_delivery_days'].fillna(df['estimated_delivery_days'].median())

X = pd.get_dummies(df.drop(columns=['is_returned']), drop_first=True)
y = df['is_returned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print("Class weights:", dict(zip(np.unique(y_train), weights)))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Balanced model
# ---------------------------------------------------------
modelbalanced = LogisticRegression(max_iter=1000, class_weight='balanced')
modelbalanced.fit(X_train_scaled, y_train)

y_predbalanced = modelbalanced.predict(X_test_scaled)
y_probbalanced = modelbalanced.predict_proba(X_test_scaled)[:, 1]

print("\nBalanced model confusion matrix:")
print(confusion_matrix(y_test, y_predbalanced))

#RocCurveDisplay.from_estimator(modelbalanced, X_test_scaled, y_test)
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
#plt.title('ROC Curve — Return Prediction (Balanced)')
#plt.legend()
#plt.show()

# ---------------------------------------------------------
# Grid search — wider C range including strong regularization
# ---------------------------------------------------------
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV F1 score:", grid.best_score_)

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nGrid search best model confusion matrix:")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

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
    'feature': X.columns,
    'coefficient': best_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 feature coefficients (grid search best model):")
print(coef_df.head(10))

# ---------------------------------------------------------
# Best threshold by F1 (only meaningful once leakage is ruled out)
# ---------------------------------------------------------
best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.05, 0.95, 0.01):
    y_pred_threshold = (y_prob_best >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest threshold: {best_threshold:.2f}, F1: {best_f1:.2f}")

y_pred_final = (y_prob_best >= best_threshold).astype(int)
print(confusion_matrix(y_test, y_pred_final))
print(f"Precision: {precision_score(y_test, y_pred_final):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred_final):.2f}")
print(f"F1:        {f1_score(y_test, y_pred_final):.2f}")

# ---------------------------------------------------------
# Comparison table
# ---------------------------------------------------------
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Returned)', 'Recall (Returned)', 'F1 (Returned)', 'ROC-AUC'],
    'Balanced': [
        accuracy_score(y_test, y_predbalanced),
        precision_score(y_test, y_predbalanced),
        recall_score(y_test, y_predbalanced),
        f1_score(y_test, y_predbalanced),
        roc_auc_score(y_test, y_probbalanced)
    ],
    'Grid Search Best': [
        accuracy_score(y_test, y_pred_best),
        precision_score(y_test, y_pred_best),
        recall_score(y_test, y_pred_best),
        f1_score(y_test, y_pred_best),
        roc_auc_score(y_test, y_prob_best)
    ]
})
comparison = comparison.round(2)
print("\n", comparison)

print("Number of records:", df.shape[0])
print("Number of model features:", X.shape[1])
print("Target variable: is_returned")