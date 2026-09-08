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
    'order_status', 'payment_status', 'loyalty_points_earned', 'discount_amount', 'loyalty_points_redeemed', 'gross_sales', 'delivery_status']
df = df.drop(columns=leak_or_useless)

# handle missing values
df['delivery_days'] = df['delivery_days'].fillna(df['delivery_days'].median())
df['estimated_delivery_days'] = df['estimated_delivery_days'].fillna(df['estimated_delivery_days'].median())

# check target balance
print(df['is_returned'].value_counts(normalize=True))

#for col in df.select_dtypes(include='object').columns:
#    if col != 'is_returned':
#        print(df.groupby(col)['is_returned'].mean())
#        print()

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

#ROC display plot
#RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
#plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
#plt.title('ROC Curve — Return Prediction')
#plt.legend()
# plt.show()
print(" ")


# quick model comparison
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Returned)', 'Recall (Returned)', 'F1 (Returned)', 'ROC-AUC'],
    'Baseline': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    ]
})

comparison = comparison.round(2)
print(comparison)

print("Number of records:", df.shape[0])
print("Number of model features:", X.shape[1])
print("Target variable: is_returned")
print(" ")
print(" ")
counts = df['is_returned'].value_counts()
percentages = df['is_returned'].value_counts(normalize=True) * 100

print("Not Returned (0):", counts[0], f"({percentages[0]:.1f}%)")
print("Returned (1):", counts[1], f"({percentages[1]:.1f}%)")

#lasso and ridge
print("Setting up L1 and L2 models...")

model_l2 = LogisticRegression(max_iter=1000, l1_ratio=0.0, class_weight='balanced', solver='saga')
model_l1 = LogisticRegression(max_iter=1000, l1_ratio=1.0, class_weight='balanced', solver='saga')

print("Fitting L1 (Lasso)...")
model_l1.fit(X_train_scaled, y_train)
print("L1 done.")

print("Fitting L2 (Ridge)...")
model_l2.fit(X_train_scaled, y_train)
print("L2 done.")

print("L1 non-zero coefficients:", (model_l1.coef_[0] != 0).sum(), "/", len(model_l1.coef_[0]))
print("L2 non-zero coefficients:", (model_l2.coef_[0] != 0).sum(), "/", len(model_l2.coef_[0]))

for c_value in [1, 0.1, 0.01, 0.001]:

    model_l1_test = LogisticRegression(
        max_iter=2000,
        l1_ratio=1.0,
        C=c_value,
        class_weight='balanced',
        solver='saga'
    )

    model_l2_test = LogisticRegression(
        max_iter=2000,
        l1_ratio=0.0,
        C=c_value,
        class_weight='balanced',
        solver='saga'
    )

    model_l1_test.fit(X_train_scaled, y_train)
    model_l2_test.fit(X_train_scaled, y_train)

    nonzero_l1 = (model_l1_test.coef_[0] != 0).sum()
    nonzero_l2 = (model_l2_test.coef_[0] != 0).sum()

    print(f"C={c_value}: L1 = {nonzero_l1}/112 non-zero coefficients")
    print(f"C={c_value}: L2 = {nonzero_l2}/112 non-zero coefficients")
