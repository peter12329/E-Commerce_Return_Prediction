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
* `delivery_status` — the `On Time`, `Delayed`, and `Early` categories had a **0% return rate**, while `Cancelled` had 38.5% — a near-perfect separator once combined with class weighting
* `loyalty_points_earned` — equal to **exactly 0 for every returned order**, with zero variance, versus a normal spread (mean ≈112) for non-returned orders
* `discount_amount` — same zero-variance pattern: exactly 0 for every returned order
* `loyalty_points_redeemed` — same zero-variance pattern: exactly 0 for every returned order

**High-cardinality categorical columns** — too many unique values to one-hot encode without exploding the feature space:
* `customer_postal_code`
* `customer_city` (15,516 unique values)

**Marketing/incentive fields deemed irrelevant to return likelihood:**
* `campaign_name`
* `coupon_code`

`gross_sales` was checked using the same distribution comparison as the leaked columns above and showed no zero-variance pattern — its values overlapped normally between returned and non-returned orders — so it was ultimately retained... *(note: confirm whether you kept or dropped `gross_sales` in your final script and adjust this line accordingly)*.

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

Removing both columns brought accuracy down to a more plausible **93.86%**, with ROC-AUC of 96.88%.

**Round 2 — a missed categorical leak (delivery status).**
`delivery_status` was checked using the same method and initially judged safe, since no single category showed a return rate of exactly 1.0. However, this missed a different form of leakage: three of its four categories (`On Time`, `Delayed`, `Early`) had a return rate of **exactly 0%**, while only `Cancelled` showed any returns (38.5%). Combined with `class_weight='balanced'` and hyperparameter tuning via `GridSearchCV`, this produced unrealistically large coefficients (magnitudes over -30) and a model that was suspiciously certain about most predictions.

**Round 3 — numeric zero-variance leakage.**
After removing `delivery_status`, coefficients remained abnormally large, now concentrated on `loyalty_points_earned` (-68.7). Grouping several numeric columns by `is_returned` revealed that `loyalty_points_earned`, `discount_amount`, and `loyalty_points_redeemed` were **all set to exactly 0, with zero variance, for every single returned order** — a downstream effect of the return itself, not a usable predictive signal.

After removing all of these columns, the model's ROC-AUC dropped to a much more modest **0.79**, and feature coefficients returned to normal, single-digit magnitudes — confirming the leakage was fully resolved.

## Machine Learning Model

### Logistic Regression

Logistic Regression was used because this is a binary classification problem.

**Input (X):**

* Customer and order-related features after preprocessing
* Categorical features converted using one-hot encoding
* **112 final model features**

**Output (Y):**

* `is_returned`

  * `0` = Not Returned
  * `1` = Returned

## Model Results (Clean, Leakage-Free)

The Logistic Regression model was evaluated using the **test dataset**, which contains **27,624 records**.

| Metric                       |  Score |
| ----------------------------- | -----: |
| **Accuracy**                  |    93% |
| **Precision (Returned)**      |    39% |
| **Recall (Returned)**         |     1% |
| **F1-Score (Returned)**       |     3% |
| **ROC-AUC**                   |    79% |

### Interpretation

Once all leaking columns were removed, the model's ROC-AUC dropped from an inflated 96-97% to a genuine **79%** — meaning the model retains real, moderate ability to rank returns above non-returns using only information that would actually be available before a return occurs.

However, at the default 0.5 classification threshold, the unweighted model almost never predicts "Returned" — catching only **1% of actual returns**. This happens because, without class weighting, the model has little incentive to flag the rare minority class, defaulting instead to the safe, common prediction ("Not Returned"). This is expected behavior for an imbalanced dataset and motivates the class-weighting approach below.

## Handling Class Imbalance

*(To complete: retrain with `class_weight='balanced'` on this fully cleaned feature set and record the results here, following the same format as above. Based on earlier experiments — before the delivery_status and numeric leaks were found — class weighting substantially raised recall at the cost of precision; confirm whether the same tradeoff holds on the clean data.)*

## Visualizations

*(Insert ROC curve and confusion matrix plots for the clean model here.)*

## Main Findings

* An initial model achieved a suspicious **100% accuracy**, traced to leakage in `order_status` and `payment_status`.
* A second round of leakage was found in `delivery_status`, whose non-`Cancelled` categories had a 0% return rate.
* A third round of leakage was found in `loyalty_points_earned`, `discount_amount`, and `loyalty_points_redeemed`, all of which were fixed at exactly 0 for every returned order.
* After removing all leaking columns, the model's true performance is **ROC-AUC 0.79**, substantially lower than the inflated 0.97 seen with leakage present — but a legitimate, defensible result.
* At the default threshold, the unweighted clean model has very low recall (1%) for returned orders, motivating further work with class weighting and/or threshold tuning.

## Limitations

* The dataset has a significant class imbalance.
* Even after removing leakage, predicting returns from order/customer metadata alone is a genuinely difficult task, as reflected in the lower ROC-AUC.
* The dataset may not represent all e-commerce customers and orders.
* Results may not generalize to other datasets or real-world situations.
* Additional features (e.g., product category, which was not merged into this dataset) and alternative machine learning algorithms could potentially improve performance.

## Conclusion

This project demonstrates how machine learning can be used to predict e-commerce returns — and, just as importantly, how easily inflated results can arise from data leakage. Three separate rounds of leakage were identified and removed over the course of the project, each caught by noticing suspiciously strong or unrealistic results. The final, leakage-free Logistic Regression model achieved a more modest but trustworthy **ROC-AUC of 0.79**, reflecting the genuine difficulty of predicting returns from information available before the fact. Future work includes retraining with `class_weight='balanced'` on the clean feature set, threshold tuning, adding relevant features such as product category, and testing other machine learning algorithms.

## Technologies

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Numpy