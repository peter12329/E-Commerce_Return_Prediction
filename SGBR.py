import importlib.util
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Script starting...")

print("Importing baseline (this re-runs the full baseline script)...")
spec = importlib.util.spec_from_file_location(
    "baseline",
    r"C:\Users\venjo\Desktop\E-Commerce Return Prediction\Linear_Logistic_Regression_Basecode.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
print("Baseline import finished.")

X_train = baseline.X_train
X_test = baseline.X_test
y_train = baseline.y_train
y_test = baseline.y_test

# ---------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------
print("Fitting Decision Tree...")
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 5},
    random_state=42
)

tree_model.fit(X_train, y_train)
print("Decision Tree fit done.")

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


cv_scores = cross_val_score(tree_model, X_train, y_train, cv=5, scoring='f1')
print(f"Cross-validated F1 scores: {cv_scores}")
print(f"Mean CV F1 score: {cv_scores.mean():.2f}")


plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

importances = tree_model.feature_importances_
features = X_train.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Feature Importances')
plt.show()