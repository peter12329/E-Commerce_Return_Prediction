import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
    'net_sales',
    'profit',
    'profit_margin_percentage',

    'delivery_status',
    'delivery_days',
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