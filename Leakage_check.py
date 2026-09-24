print("Script starting...")
import importlib.util
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

print("Importing baseline (this re-runs the full baseline script)...")
spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\Linear_Logistic_Regression_Basecode.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
print("Baseline import finished.")

df = baseline.df  # cleaned dataframe (leaky/useless columns already dropped)

print("\nColumns being checked:", list(df.columns.drop('is_returned')))

# ---------------------------------------------------------
# Check 1: how well does each single column predict the target?
# ---------------------------------------------------------
print("\nScoring each column on its own (depth-4 tree, 3-fold CV ROC-AUC)...")
scores = {}
for col in df.columns.drop('is_returned'):
    Xc = pd.get_dummies(df[[col]], drop_first=True).fillna(-1)
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    scores[col] = cross_val_score(tree, Xc, df['is_returned'], cv=3, scoring='roc_auc').mean()
    print(f"  {col}: {scores[col]:.3f}")

print("\nTop 10 single-column ROC-AUC (0.5 = useless, >0.85 = suspicious):")
print(pd.Series(scores).sort_values(ascending=False).head(10))

# ---------------------------------------------------------
# Check 2: does missingness differ between returned / not returned?
# ---------------------------------------------------------
print("\nMissing-value rate by target (big gap = red flag):")
miss = df.isna().groupby(df['is_returned']).mean().T
miss['gap'] = (miss[1] - miss[0]).abs()
print(miss.sort_values('gap', ascending=False).head(10))

print("\nDone.")


print("\nDoes margin = profit / net_sales * 100 for every row?")
implied = df['profit'] / df['net_sales'] * 100
diff = (implied - df['profit_margin_percentage']).abs()
print(diff.groupby(df['is_returned']).describe())

print("\nAblation: decision tree with vs without profit_margin_percentage")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
for name, cols in [("with margin", baseline.X_train.columns),
                   ("without margin", baseline.X_train.columns.drop('profit_margin_percentage'))]:
    t = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
    t.fit(baseline.X_train[cols], baseline.y_train)
    print(f"  {name}: ROC-AUC = {roc_auc_score(baseline.y_test, t.predict_proba(baseline.X_test[cols])[:, 1]):.3f}")



print("")
print("")
print("")
print("\nprofit_margin_percentage by target:")
print(df.groupby('is_returned')['profit_margin_percentage'].describe())

print("\nReturn rate by margin decile:")
buckets = pd.qcut(df['profit_margin_percentage'], 10, duplicates='drop')
print(df.groupby(buckets, observed=True)['is_returned'].mean())

