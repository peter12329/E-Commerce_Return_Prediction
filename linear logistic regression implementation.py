import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
import numpy as np

path = r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\datasets\ecommerce_sales_customer_analytics_150k.csv"
df = pd.read_csv(path)

# Y variable
df['is_returned'] = df['return_status'].notna().astype(int)

# drop useless info
leak_or_useless = [
    'order_id', 'customer_id', 'customer_name', 'order_date', 'order_time',
    'return_status', 'return_reason', 'customer_review', 'review_sentiment',
    'customer_rating', 'campaign_name', 'coupon_code',
    'customer_postal_code', 'customer_city',
    'order_status', 'payment_status']
df = df.drop(columns=leak_or_useless)

# handle missing values
df['delivery_days'] = df['delivery_days'].fillna(df['delivery_days'].median())
df['estimated_delivery_days'] = df['estimated_delivery_days'].fillna(df['estimated_delivery_days'].median())

# check target balance
print(df['is_returned'].value_counts(normalize=True))

for col in df.select_dtypes(include='object').columns:
    if col != 'is_returned':
        print(df.groupby(col)['is_returned'].mean())
        print()

# dummies
X = pd.get_dummies(df.drop(columns=['is_returned']), drop_first=True)
y = df['is_returned']


# t/t split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print(dict(zip(np.unique(y_train), weights)))

# scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print(confusion_matrix(y_test, y_pred))

# ROC unbalancedcurve
RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
#plt.title('ROC Curve — Return Prediction')
#plt.legend()
# plt.show()

print(" ")

#balanced version
modelbalanced = LogisticRegression(max_iter=1000, class_weight='balanced')
modelbalanced.fit(X_train_scaled, y_train)

y_predbalanced = modelbalanced.predict(X_test_scaled)
y_probbalanced = modelbalanced.predict_proba(X_test_scaled)[:, 1]
print(confusion_matrix(y_test, y_predbalanced))

RocCurveDisplay.from_estimator(modelbalanced, X_test_scaled, y_test)
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
#plt.title('ROC Curve — Return Prediction (balanced)')
#plt.legend()
# plt.show()

#grid search for hyperparameters
param_grid = {
    'C': [10, 100, 1000, 10000],
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

best_model = grid.best_estimator_
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
#threshold = 0.95
#y_pred_best = (y_prob_best >= threshold).astype(int)

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    y_pred_threshold = (y_prob_best >= threshold).astype(int)

    print(" ")
    print(f"\nThreshold: {threshold}")
    print(confusion_matrix(y_test, y_pred_threshold))
    print(f"Precision: {precision_score(y_test, y_pred_threshold):.2f}")
    print(f"Recall:    {recall_score(y_test, y_pred_threshold):.2f}")
    print(f"F1:        {f1_score(y_test, y_pred_threshold):.2f}")


print(" ")
print(" ")
print(" ")

print("Baseline:")
print(confusion_matrix(y_test, y_pred))

print("\nBalanced:")
print(confusion_matrix(y_test, y_predbalanced))

#print("\nGrid Search:")
#print(confusion_matrix(y_test, y_pred_best))
#print("")

# quick model comparison
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Returned)', 'Recall (Returned)', 'F1 (Returned)', 'ROC-AUC'],
    'Baseline': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    ],
    'Balanced': [
        accuracy_score(y_test, y_predbalanced),
        precision_score(y_test, y_predbalanced),
        recall_score(y_test, y_predbalanced),
        f1_score(y_test, y_predbalanced),
        roc_auc_score(y_test, y_probbalanced)
    ]
        #'Grid Search Best': [
        #accuracy_score(y_test, y_pred_best),
        #precision_score(y_test, y_pred_best),
        #recall_score(y_test, y_pred_best),
        #f1_score(y_test, y_pred_best),
        #roc_auc_score(y_test, y_prob_best)
    #]
})
comparison = comparison.round(2)
print(comparison)