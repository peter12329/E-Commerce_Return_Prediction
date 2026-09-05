# E-Commerce Return Prediction

## About

This project uses machine learning to predict whether an e-commerce order will be returned.

## Objective

To build a machine learning model that can predict product returns based on order and customer information.

## Dataset

* **Number of records:** 138,116
* **Number of model features:** 112
* **Target variable:** `is_returned`
* **Dataset source:** [Kaggle](https://www.kaggle.com/datasets/datascikhan/e-commerce-sales-and-customer-analytics?select=ecommerce_sales_customer_analytics_150k.csv)

### Data Preparation

The original dataset was obtained from Kaggle. Before training the model, the dataset was inspected for missing values, duplicate records, data types, and other potential issues.

### Excluded Columns

Columns were excluded for three distinct reasons:

**Identifiers / free text (no predictive value):**
* `order_id`
* `customer_id`
* `customer_name`
* `order_date`
* `order_time`
* `customer_review`

**Post-return / target leakage** — these columns are only populated, determined, or reset *after* a return occurs, so including them would let the model "see" the answer rather than genuinely predict it:
* `return_status` (used to build the target itself)
* `return_reason`
* `review_sentiment`
* `customer_rating`
* `order_status` — the `Returned` category aligned perfectly with `is_returned = 1`
* `payment_status` — the `Refunded` category aligned perfectly with `is_returned = 1`
* `delivery_status` — the `On Time`, `Delayed`, and `Early` categories had a **0% return rate**, while `Cancelled` had 38.5%
* `loyalty_points_earned` — equal to **exactly 0 for every returned order**, with zero variance
* `discount_amount` — same zero-variance pattern: exactly 0 for every returned order
* `loyalty_points_redeemed` — same zero-variance pattern: exactly 0 for every returned order

**High-cardinality categorical columns** — too many unique values to one-hot encode without exploding the feature space:
* `customer_postal_code`
* `customer_city` (15,516 unique values)

**Marketing/incentive fields deemed irrelevant to return likelihood:**
* `campaign_name`
* `coupon_code`

`gross_sales` was checked using the same distribution comparison used to detect the numeric leaks above, and showed no zero-variance pattern (its values overlapped normally between returned and non-returned orders), so it was retained.

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

### Missing Values

Missing values in `delivery_days` and `estimated_delivery_days` were handled using the median value of each respective column. After preprocessing, both columns contained **0 missing values**.

### Target Distribution

| Target           |   Count | Percentage |
| ---------------- | ------: | ---------: |
| Not Returned (0) | 128,654 |      93.1% |
| Returned (1)     |   9,462 |       6.9% |

## Data Preprocessing

1. Created the binary target variable `is_returned`.
2. Removed identifier, post-return leakage, high-cardinality, and irrelevant columns (see Excluded Columns above).
3. Filled missing values in `delivery_days` and `estimated_delivery_days` using the median.
4. Converted categorical variables using one-hot encoding.
5. Split the data into **80% training** and **20% testing** data.
6. Standardized the features using `StandardScaler`.

After one-hot encoding, the dataset contained **112 model features**.

## Leakage Detection

Leakage was found and removed across **three separate rounds** of investigation, each triggered by suspiciously strong results.

**Round 1 — categorical leakage (order/payment status).**
An initial model, trained before any leakage checks, achieved **100% accuracy** on the test set — a strong sign of leakage in an imbalanced, real-world classification task. Checking the mean return rate per category for every categorical column revealed:
* `order_status = "Returned"` had a mean `is_returned` of **1.0** (all other categories, 0.0)
* `payment_status = "Refunded"` had a mean `is_returned` of **1.0** (all other categories, 0.0)

Removing both columns brought accuracy down to a more plausible 93.86%, with ROC-AUC of 96.88% — still too high to be trustworthy, as later rounds confirmed.

**Round 2 — a missed categorical leak (delivery status).**
`delivery_status` was checked using the same method and initially judged safe, since no single category showed a return rate of exactly 1.0. This missed a different form of leakage: three of its four categories (`On Time`, `Delayed`, `Early`) had a return rate of **exactly 0%**, while only `Cancelled` showed any returns (38.5%). Combined with `class_weight='balanced'` and hyperparameter tuning via `GridSearchCV`, this produced unrealistically large coefficients (magnitudes over -30) and near-perfect recall.

**Round 3 — numeric zero-variance leakage.**
After removing `delivery_status`, coefficients remained abnormally large, now concentrated on `loyalty_points_earned` (-68.7). Grouping several numeric columns by `is_returned` revealed that `loyalty_points_earned`, `discount_amount`, and `loyalty_points_redeemed` were **all set to exactly 0, with zero variance, for every single returned order** — a downstream effect of the return itself, not a usable predictive signal.

After removing all leaking columns, ROC-AUC dropped to a genuine **0.79**, and feature coefficients returned to normal, single-digit magnitudes, confirming the leakage was fully resolved.

## Machine Learning Model

### Logistic Regression

Logistic Regression was used because this is a binary classification problem. Three variants were trained and compared on the same clean, leakage-free feature set:

1. **Baseline** — default `LogisticRegression`, no class weighting
2. **Balanced** — `class_weight='balanced'` to counter the 93/7 class imbalance
3. **Grid Search** — `GridSearchCV` tuning `C` (regularization strength) and `class_weight`, optimizing for F1-score via 5-fold cross-validation

**Input (X):**

* Customer and order-related features after preprocessing
* Categorical features converted using one-hot encoding
* **112 final model features**

**Output (Y):**

* `is_returned`

  * `0` = Not Returned
  * `1` = Returned

## Model Comparison (Clean, Leakage-Free Data)

The models were evaluated on the **test dataset**, which contains **27,624 records**.

| Metric                    | Baseline | Balanced | Grid Search |
| -------------------------- | -------: | -------: | ----------: |
| **Accuracy**               |     0.93 |     0.69 |        0.69 |
| **Precision (Returned)**   |     0.39 |     0.15 |        0.15 |
| **Recall (Returned)**      |     0.01 |     0.76 |        0.76 |
| **F1-Score (Returned)**    |     0.03 |     0.25 |        0.25 |
| **ROC-AUC**                |     0.79 |     0.79 |        0.79 |

**Confusion Matrices:**

Baseline:
```
[[25693    39]
 [ 1867    25]]
```

Balanced:
```
[[17589  8143]
 [  454  1438]]
```

Grid Search Best (`C=10`, `class_weight='balanced'`):
```
[[17588  8144]
 [  454  1438]]
```

### Interpretation

All three models share the same ROC-AUC (0.79), confirming they have identical underlying discriminative ability — `class_weight='balanced'` and hyperparameter tuning don't change what the model *can* distinguish, only where it draws the decision boundary.

* **Baseline** is heavily biased toward the majority class: it predicts "Returned" almost never (1% recall), making it accurate overall (93%) but practically useless for catching actual returns.
* **Balanced** and **Grid Search** produce nearly identical results — both trade a large amount of precision (0.15) for much higher recall (0.76), catching roughly three-quarters of actual returns at the cost of many false alarms. Grid search confirmed that `class_weight='balanced'` combined with `C=10` was already close to optimal (best CV F1 score: 0.246) — it did not find a meaningfully better configuration than manually setting `class_weight='balanced'`.
* **Threshold tuning** on the grid search model, by scanning thresholds to maximize F1, found an optimal threshold of **0.69**, yielding a more moderate tradeoff: Precision 0.19, Recall 0.43, F1 0.27 — a middle ground between the extremes of the baseline and the balanced/grid-search models.

This confirms that, on genuinely leakage-free data, predicting e-commerce returns from order and customer metadata alone is a difficult problem: even the best-tuned logistic regression achieves modest results (F1 ≈ 0.25–0.27 for the minority class), reflecting real limits in the available signal rather than a modeling shortcoming.

## Visualizations

*(Insert ROC curve and confusion matrix plots for the clean models here.)*

## Main Findings

* An initial model achieved a suspicious **100% accuracy**, traced to leakage in `order_status` and `payment_status`.
* A second round of leakage was found in `delivery_status`, whose non-`Cancelled` categories had a 0% return rate.
* A third round of leakage was found in `loyalty_points_earned`, `discount_amount`, and `loyalty_points_redeemed`, all of which were fixed at exactly 0 for every returned order.
* After removing all leaking columns, the model's true performance is **ROC-AUC 0.79** — substantially lower than the inflated 0.97 seen with leakage present, but a legitimate, defensible result.
* The unweighted baseline has extremely low recall (1%) for returned orders; `class_weight='balanced'` raises recall to 76% at the cost of precision dropping to 15%.
* `GridSearchCV` confirmed `class_weight='balanced'` with `C=10` as near-optimal — tuning did not meaningfully outperform manual class weighting.
* Threshold tuning offers a middle-ground option (Precision 0.19, Recall 0.43, F1 0.27) between the aggressive recall-focused balanced model and the overly conservative baseline.

## Limitations

* The dataset has a significant class imbalance.
* Even after removing leakage, predicting returns from order/customer metadata alone is a genuinely difficult task, as reflected in the ROC-AUC of 0.79 and modest F1-scores across all model variants.
* The dataset may not represent all e-commerce customers and orders.
* Results may not generalize to other datasets or real-world situations.
* Additional features (e.g., product category, which was not merged into this dataset) and alternative machine learning algorithms could potentially improve performance.

## Conclusion

This project demonstrates how machine learning can be used to predict e-commerce returns — and, just as importantly, how easily inflated results can arise from data leakage. Three separate rounds of leakage were identified and removed over the course of the project, each caught by noticing suspiciously strong or unrealistic results (100% accuracy, near-perfect recall, and abnormally large model coefficients).

The final, leakage-free Logistic Regression models achieved a modest but trustworthy **ROC-AUC of 0.79** across all variants. Class weighting and hyperparameter tuning both meaningfully improved the model's ability to catch actual returns (recall rising from 1% to 76%), at the cost of more false alarms (precision falling to 15%) — a genuine precision/recall tradeoff rather than a straightforward improvement. Threshold tuning offers a way to land at a different point along that same tradeoff.

Future work includes adding relevant features such as product category, exploring resampling techniques (e.g. SMOTE) as an alternative to class weighting, and testing other machine learning algorithms.

## Technologies

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Numpy