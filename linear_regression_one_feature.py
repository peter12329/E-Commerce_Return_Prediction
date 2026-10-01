import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

path = r"C:\Users\Test\data_science\E-Commerce_Return_Prediction\datasets\ecommerce_sales_customer_analytics_150k.csv"

df = pd.read_csv(path)

# ---------------------------------------------------------
# Y variable
# ---------------------------------------------------------

df['is_returned'] = df['return_status'].notna().astype(int)

# ---------------------------------------------------------
# Drop useless information / leakage
# ---------------------------------------------------------

leak_or_useless = [
    'return_status', 'return_reason',
    'customer_review', 'review_sentiment', 'customer_rating',
    'order_status', 'payment_status',
    'delivery_status', 'delivery_days',
    'order_id', 'customer_id', 'customer_name',
    'customer_postal_code', 'customer_city',
    'order_date', 'order_time',
    'campaign_name', 'coupon_code',
    'discount_amount', 'gross_sales',
    'loyalty_points_earned', 'loyalty_points_redeemed',
    'estimated_delivery_days'
]

df = df.drop(columns=leak_or_useless)

# ---------------------------------------------------------
# Single feature
# ---------------------------------------------------------

X = df[['profit_margin_percentage']]
y = df['is_returned']

# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# Scale the single feature
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------

single_feature_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

single_feature_model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred_single = single_feature_model.predict(X_test_scaled)

y_prob_single = single_feature_model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

print(confusion_matrix(y_test, y_pred_single))

print(f"Accuracy:  {accuracy_score(y_test, y_pred_single):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_single):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred_single):.2f}")
print(f"F1:        {f1_score(y_test, y_pred_single):.2f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_single):.2f}")