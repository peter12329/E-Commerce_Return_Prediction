# E-Commerce Return Prediction

## About

This project uses machine learning to predict whether an e-commerce order will be returned.

## Objective

To build a machine learning model that can predict product returns based on order and customer information.

## Dataset

* **Number of records:** 138,116
* **Target variable:** `is_returned`
* **Dataset source:** [Kaggle](https://www.kaggle.com/datasets/datascikhan/e-commerce-sales-and-customer-analytics?select=ecommerce_sales_customer_analytics_150k.csv)

### Data Preparation

The original dataset was obtained from Kaggle. Before training the model, the dataset was inspected for missing values, duplicate records, data types, and other potential issues.

### Excluded Columns

Columns were excluded for four distinct reasons:

**Target source** — used only to construct the label itself:
* `return_status`
* `return_reason`

**Outcome leakage** — only known, populated, or determined *after* the order's outcome (a return) occurs, so including them would let the model "see" the answer rather than genuinely predict it:
* `customer_review`
* `review_sentiment`
* `customer_rating`
* `order_status` — the `Returned` category aligned perfectly with `is_returned = 1`
* `payment_status` — the `Refunded` category aligned perfectly with `is_returned = 1`
* `delivery_status` — the `On Time`, `Delayed`, and `Early` categories had a **0% return rate**, while `Cancelled` had 38.5%
* `delivery_days` — fixed at **exactly 4, with zero variance**, for every single returned order

**Identifiers / high-cardinality columns:**
* `order_id`, `customer_id`, `customer_name` — unique identifiers with no predictive value
* `customer_postal_code`, `customer_city` (15,516 unique values) — too many categories to one-hot encode without exploding the feature space

**Dropped as a conservative choice (not confirmed leaks):**
* `order_date`, `order_time`
* `campaign_name`, `coupon_code`
* `discount_amount`
* `gross_sales`
* `loyalty_points_earned`, `loyalty_points_redeemed`
* `estimated_delivery_days`

Note: `discount_amount`, `loyalty_points_earned`, and `loyalty_points_redeemed` were also found, during an earlier investigation round, to be fixed at exactly 0 with zero variance for every returned order — the same deterministic pattern confirmed as leakage elsewhere in this project. They are grouped here as a conservative exclusion rather than re-litigated as confirmed leaks, but the same evidence applies to them as to the outcome-leakage columns above.

**A false-positive check.** `net_sales`, `profit`, and `profit_margin_percentage` were briefly suspected of leakage and excluded on a precautionary basis. A groupby comparison against `is_returned` showed no zero-variance pattern for any of the three — their distributions overlapped normally between returned and non-returned orders, with `profit_margin_percentage` in particular showing a real (non-deterministic) difference in means (44.4% vs. 54.1%). All three were confirmed clean and restored to the feature set; removing them had dropped ROC-AUC from 0.79 to 0.63, and restoring them recovered it to 0.78 — direct evidence that this had been real, legitimate signal, not leakage.

### Target Variable

The target variable `is_returned` was created from `return_status`.

| Value | Meaning      |
| ----- | ------------ |
| 0     | Not Returned |
| 1     | Returned     |

## Initial Data Findings

Before training the model, the following observations were found:

* The dataset contains **138,116 records**.
* **128,654 (93.1%)** orders were not returned.
* **9,462 (6.9%)** orders were returned.
* The target variable `is_returned` is **imbalanced**, with significantly more non-returned orders than returned orders.

### Target Distribution

| Target           |   Count | Percentage |
| ---------------- | ------: | ---------: |
| Not Returned (0) | 128,654 |      93.1% |
| Returned (1)     |   9,462 |       6.9% |

## Data Preprocessing

1. Created the binary target variable `is_returned`.
2. Removed target-source, outcome-leakage, identifier/high-cardinality, and conservatively-excluded columns (see Excluded Columns above).
3. Converted categorical variables using one-hot encoding.
4. Split the data into **80% training** and **20% testing** data.
5. Standardized the features using `StandardScaler` (for the linear/SVM models; tree-based models used unscaled features).

## Leakage Detection

Leakage was found and removed across **three separate rounds** of investigation, each triggered by suspiciously strong results, plus a fourth round that ruled out a false positive.

**Round 1 — categorical leakage (order/payment status).**
An initial model, trained before any leakage checks, achieved **100% accuracy** on the test set — a strong sign of leakage in an imbalanced, real-world classification task. Checking the mean return rate per category for every categorical column revealed:
* `order_status = "Returned"` had a mean `is_returned` of **1.0** (all other categories, 0.0)
* `payment_status = "Refunded"` had a mean `is_returned` of **1.0** (all other categories, 0.0)

Removing both columns brought accuracy down to a more plausible 93.86%, with ROC-AUC of 96.88% — still too high to be trustworthy, as later rounds confirmed.

**Round 2 — a missed categorical leak (delivery status).**
`delivery_status` was checked using the same method and initially judged safe, since no single category showed a return rate of exactly 1.0. This missed a different form of leakage: three of its four categories (`On Time`, `Delayed`, `Early`) had a return rate of **exactly 0%**, while only `Cancelled` showed any returns (38.5%). Combined with `class_weight='balanced'` and hyperparameter tuning via `GridSearchCV`, this produced unrealistically large coefficients (magnitudes over -30) and near-perfect recall.

**Round 3 — numeric zero-variance leakage.**
After removing `delivery_status`, coefficients remained abnormally large, now concentrated on `loyalty_points_earned` (-68.7). Grouping several numeric columns by `is_returned` revealed that `loyalty_points_earned`, `discount_amount`, `loyalty_points_redeemed`, `delivery_days`, and `estimated_delivery_days` were **all fixed at a single value, with zero variance, for every returned order** — a downstream effect of the return itself, not a usable predictive signal.

**Round 4 — a false positive, caught and corrected.**
`net_sales`, `profit`, and `profit_margin_percentage` were excluded as a precaution without being confirmed as leaks. Their exclusion caused ROC-AUC to fall to 0.63 — a meaningful, suspicious drop. A groupby check showed no zero-variance pattern for any of the three; they were restored, and ROC-AUC recovered to 0.78. This round demonstrated that the same diagnostic method used to catch leakage can also catch mistaken exclusions.

After finalizing the feature set, ROC-AUC settled at a genuine **~0.78**, and feature coefficients returned to normal, single-digit magnitudes, confirming the leakage was resolved without discarding legitimate signal.

## Machine Learning Models

Eight approaches were trained and compared on the same clean feature set:

1. **Baseline** — default `LogisticRegression`, no class weighting
2. **Balanced** — `LogisticRegression` with `class_weight='balanced'`
3. **Grid Search** — `GridSearchCV` tuning `C` and `class_weight`, optimizing for F1-score via 5-fold cross-validation
4. **SVM** — `LinearSVC` wrapped in `CalibratedClassifierCV` for probability estimates, `class_weight='balanced'`
5. **Decision Tree** — `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight={0: 1, 1: 5}`
6. **L1 (Lasso)** — `LogisticRegression` with `l1_ratio=1.0`, `solver='saga'`, `class_weight='balanced'`, `C=1`
7. **Bagging** — `BaggingClassifier`, 100 trees, out-of-bag (OOB) scoring enabled
8. **Random Forest** — `RandomForestClassifier`, default class weighting

**Output (Y):** `is_returned` (`0` = Not Returned, `1` = Returned)

## Model Comparison

The models were evaluated on the **test dataset** (27,624 records).

| Metric                  | Baseline | Balanced | Grid Search | SVM  | Decision Tree | L1 (C=1) | Bagging (100 trees) | Random Forest |
| ------------------------ | -------: | -------: | ----------: | ---: | -------------: | -------: | -------------------: | -------------: |
| **Accuracy**              |     0.93 |     0.68 |        0.68 | 0.93 |           0.89 |      N/A |                   N/A |           0.62 |
| **Precision (Returned)**  |     0.42 |     0.14 |        0.14 | 0.39 |           0.21 |     0.14 |                  0.23 |           0.13 |
| **Recall (Returned)**     |     0.01 |     0.74 |        0.74 | 0.01 |           0.23 |     0.74 |                  0.00 |           0.78 |
| **F1-Score (Returned)**   |     0.02 |     0.24 |        0.24 | 0.02 |           0.22 |     0.24 |                  0.00 |           0.22 |
| **ROC-AUC**               |     0.78 |     0.78 |         N/A | 0.78 |           0.76 |     0.78 |                  0.75 |           0.76 |

**Confusion Matrices:**

Baseline:
```
[[25699    33]
 [ 1868    24]]
```

Balanced:
```
[[17420  8312]
 [  486  1406]]
```

Grid Search Best (`C=10`, `class_weight='balanced'`, best CV F1 = 0.238):
```
[[17420  8312]
 [  486  1406]]
```

SVM:
```
[[25697    35]
 [ 1870    22]]
```

Decision Tree:
```
[[24105  1627]
 [ 1461   431]]
```

Bagging (100 trees, OOB score 0.9310):
```
(recall collapsed to 0.00 — see Interpretation below)
```

Random Forest:
```
[[15725 10007]
 [  414  1478]]
```

### Random Forest — Top 10 Feature Importances

| Feature                    | Importance |
| --------------------------- | ---------: |
| `profit_margin_percentage`  |     0.4793 |
| `profit`                    |     0.1218 |
| `product_cost`               |     0.0773 |
| `quantity`                   |     0.0595 |
| `net_sales`                  |     0.0486 |
| `tax_amount`                 |     0.0358 |
| `shipping_cost`              |     0.0199 |
| `customer_lifetime_value`    |     0.0194 |
| `customer_age`               |     0.0147 |
| `customer_order_count`       |     0.0097 |

### Interpretation

* **Baseline and SVM behave almost identically** — both are heavily biased toward the majority class, achieving high accuracy (0.93) purely by predicting "Not Returned" almost every time (Recall 0.01). This makes both practically useless for the actual goal of catching returns.
* **Balanced, Grid Search, and L1 converge on the same result** (Precision 0.14, Recall 0.74, F1 0.24, ROC-AUC 0.78). This is a strong signal that class weighting — not the specific algorithm variant — is what drives the recall improvement. Grid search's best parameters (`C=10`, `class_weight='balanced'`) essentially rediscovered manual balancing rather than finding a meaningfully better configuration.
* **Random Forest** achieves the highest recall (0.78) but the lowest accuracy (0.62) of any model — the most aggressive at flagging returns, and correspondingly the most prone to false alarms. `profit_margin_percentage` alone accounts for nearly half its feature importance (0.4793), consistent with the same feature standing out in the false-positive check above.
* **Bagging's recall collapsed to 0.00** despite a respectable out-of-bag accuracy score (0.9310). This is very likely because `BaggingClassifier`'s default base estimator (a plain decision tree) was not given any class weighting, so — like the unweighted baseline — it defaulted to predicting the majority class almost exclusively. This is flagged as a likely fix rather than a finding: rerunning with a class-weighted base estimator would be expected to bring its recall in line with the other weighted models.
* **Decision Tree** sits in a weaker middle ground on both precision and recall, likely because its manually chosen `class_weight={0: 1, 1: 5}` is considerably gentler than the `'balanced'` setting's actual computed ratio (roughly 13.6:1).

This confirms that, on genuinely leakage-free data, predicting e-commerce returns from order and customer metadata alone is a difficult problem: even across several different model families, no configuration achieves better than a modest F1 (~0.22–0.24) for the minority class, reflecting real limits in the available signal rather than a modeling shortcoming.

## Visualizations

*(Insert ROC curve, confusion matrix, and feature importance plots here.)*

## Main Findings

* An initial model achieved a suspicious **100% accuracy**, traced to leakage in `order_status` and `payment_status`.
* A second round of leakage was found in `delivery_status`, whose non-`Cancelled` categories had a 0% return rate.
* A third round of leakage was found in `loyalty_points_earned`, `discount_amount`, `loyalty_points_redeemed`, `delivery_days`, and `estimated_delivery_days`, all fixed at a single value with zero variance for every returned order.
* A fourth investigation caught the opposite mistake: `net_sales`, `profit`, and `profit_margin_percentage` had been excluded as a false-positive precaution, dropping ROC-AUC from 0.79 to 0.63; restoring them recovered ROC-AUC to 0.78.
* Across eight different models, class weighting (however it is applied — `'balanced'`, grid search, or L1) consistently raises recall from ~1% to ~74–78%, at the cost of precision falling to ~0.13–0.15 — a genuine, unavoidable tradeoff rather than a free improvement.
* `profit_margin_percentage` is the single most important feature by a wide margin (47.9% of Random Forest's feature importance), consistent with its role in the Round 4 false-positive check.
* Bagging's near-zero recall is most plausibly an artifact of its base estimator not being class-weighted, rather than a genuine limitation of the bagging approach itself.

## Limitations

* The dataset has a significant class imbalance.
* Even after resolving leakage, predicting returns from order/customer metadata alone is a genuinely difficult task, as reflected in ROC-AUC values in the 0.75–0.78 range and modest F1-scores across all model variants.
* The Bagging result is likely confounded by a missing class-weight setting on its base estimator, and should be re-evaluated before being treated as a fair comparison point.
* The dataset may not represent all e-commerce customers and orders.
* Results may not generalize to other datasets or real-world situations.
* Additional features (e.g., product category, which was not merged into this dataset) could potentially improve performance further.

## Conclusion

This project demonstrates how machine learning can be used to predict e-commerce returns — and, just as importantly, how easily inflated or deflated results can arise from mistakes in feature selection. Four separate rounds of investigation were carried out over the course of the project: three uncovered genuine data leakage, and a fourth caught an overly cautious exclusion that had discarded real predictive signal.

Across eight model variants — logistic regression (baseline, balanced, grid-searched, and L1-regularized), SVM, a decision tree, bagging, and a random forest — the final, leakage-free results converge on a consistent story: ROC-AUC in the 0.75–0.78 range represents a real, defensible ceiling for this feature set, and class weighting is the single most effective lever for trading precision for recall, regardless of which underlying algorithm is used. `profit_margin_percentage` stands out as the most informative individual feature across models.

Future work includes re-running Bagging with a class-weighted base estimator, adding relevant features such as product category, exploring resampling techniques (e.g. SMOTE) as an alternative to class weighting, and further tuning Random Forest and Decision Tree depth/leaf parameters to reduce their false-positive rates.

## Technologies

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Numpy