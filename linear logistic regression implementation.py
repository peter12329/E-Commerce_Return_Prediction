import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.utils.class_weight import compute_class_weight
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

# prediction
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

y_prob = model.predict_proba(X_test_scaled)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# ROC curve
RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.title('ROC Curve — Return Prediction')
plt.legend()
plt.show()

print(" ")

#balanced version
modelbalanced = LogisticRegression(max_iter=1000, class_weight='balanced')
modelbalanced.fit(X_train_scaled, y_train)

y_predbalanced = modelbalanced.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_predbalanced))
print(classification_report(y_test, y_predbalanced))
print(confusion_matrix(y_test, y_predbalanced))

y_probbalanced = modelbalanced.predict_proba(X_test_scaled)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_predbalanced))

RocCurveDisplay.from_estimator(modelbalanced, X_test_scaled, y_test)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.title('ROC Curve — Return Prediction(balanced)')
plt.legend()
plt.show()