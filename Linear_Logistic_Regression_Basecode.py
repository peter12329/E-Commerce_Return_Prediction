import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


path = r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\datasets\ecommerce_sales_customer_analytics_150k.csv"
df = pd.read_csv(path)

# ---------------------------------------------------------
# Y variable
# ---------------------------------------------------------

df['is_returned'] = df['return_status'].notna().astype(int)

# ---------------------------------------------------------
# drop useless info
# ---------------------------------------------------------

leak_or_useless = [
    # target source
    'return_status', 'return_reason',

    # only known after the order (outcome leaks)
    'customer_review', 'review_sentiment', 'customer_rating',
    'order_status', 'payment_status',
    'delivery_status', 'delivery_days',

    # identifiers / high-cardinality
    'order_id', 'customer_id', 'customer_name',
    'customer_postal_code', 'customer_city',

    # dropped as a conservative choice (not leaks)
    'order_date', 'order_time',
    'campaign_name', 'coupon_code',
    'discount_amount', 'gross_sales',
    'loyalty_points_earned', 'loyalty_points_redeemed',
    'estimated_delivery_days'
]
df = df.drop(columns=leak_or_useless)

# ---------------------------------------------------------
# dummies
# ---------------------------------------------------------

X = pd.get_dummies(df.drop(columns=['is_returned']), drop_first=True)
y = df['is_returned']

# ---------------------------------------------------------
# train test split split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
#model
# ---------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(" ")
print("\nBaseline")
print(confusion_matrix(y_test, y_pred))
 
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred):.2f}")
print(f"F1:        {f1_score(y_test, y_pred):.2f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.2f}")
print("Baseline script finished.")
