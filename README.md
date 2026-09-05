# E-Commerce Return Prediction

## About

This project uses machine learning to predict whether an e-commerce order will be returned.

## Objective

To build a machine learning model that can predict product returns based on order and customer information.

## Dataset

* **Number of records:** 138,116
* **Number of model features:** 119
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

**Post-return / target leakage** — these columns are only populated or determined *after* a return occurs, so including them would let the model "see" the answer:
* `return_status` (used to build the target itself)
* `return_reason`
* `review_sentiment`
* `customer_rating`
* `order_status` — one category (`Returned`) was found to align perfectly with `is_returned = 1`
* `payment_status` — one category (`Refunded`) was found to align perfectly with `is_returned = 1`

**High-cardinality categorical columns** — too many unique values to one-hot encode without exploding the feature space:
* `customer_postal_code`
* `customer_city` (15,516 unique values)

**Marketing/incentive fields deemed irrelevant to return likelihood:**
* `campaign_name`
* `coupon_code`

`delivery_status` was tested for leakage the same way as `order_status` and `payment_status` (checking the mean return rate per category). It showed no near-perfect split — its highest category (`Cancelled`) had a 38.5% return rate, with all other categories near 0% — so it was retained as a legitimate feature rather than dropped.

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
2. Removed identifier, post-return, high-cardinality, and irrelevant columns (see Excluded Columns above).
3. Filled missing values in `delivery_days` and `estimated_delivery_days` using the median.
4. Converted categorical variables using one-hot encoding.
5. Split the data into **80% training** and **20% testing** data.
6. Standardized the features using `StandardScaler`.

After one-hot encoding, the dataset contained **119 model features**.

## Leakage Detection

An initial version of the model, trained before `order_status` and `payment_status` were removed, achieved **100% accuracy** on the test set. A perfect score on a real-world imbalanced classification task is a strong sign of data leakage rather than genuine model skill.

To find the cause, the mean return rate was computed for every category within each categorical column:

```python
for col in df.select_dtypes(include='object').columns:
    print(df.groupby(col)['is_returned'].mean())
```

This showed that:
* `order_status = "Returned"` had a mean `is_returned` of **1.0** (and all other categories, 0.0)
* `payment_status = "Refunded"` had a mean `is_returned` of **1.0** (and all other categories, 0.0)

Both columns were removed, after which the model's accuracy dropped to a realistic **93.86%**, confirming the leak was resolved.

## Machine Learning Model

### Logistic Regression

Logistic Regression was used because this is a binary classification problem.

**Input (X):**

* Customer and order-related features after preprocessing
* Categorical features converted using one-hot encoding
* **119 final model features**

**Output (Y):**

* `is_returned`

  * `0` = Not Returned
  * `1` = Returned

## Model Results

The Logistic Regression model was evaluated using the **test dataset**, which contains **27,624 records**.

| Metric                       |  Score |
| ----------------------------- | -----: |
| **Accuracy**                  | 93.86% |
| **Precision (Not Returned)**  |    97% |
| **Recall (Not Returned)**     |    97% |
| **F1-Score (Not Returned)**   |    97% |
| **Precision (Returned)**      |    55% |
| **Recall (Returned)**         |    56% |
| **F1-Score (Returned)**       |    56% |
| **ROC-AUC**                   | 96.88% |

### Confusion Matrix

| Actual / Predicted   | Not Returned (0) | Returned (1) |
| --------------------- | ---------------: | -----------: |
| **Not Returned (0)**  |           24,869 |          863 |
| **Returned (1)**      |              833 |        1,059 |

### ROC-AUC

The model achieved a **ROC-AUC score of 96.88%**, indicating a strong ability to *rank* returned orders above non-returned orders across all thresholds. This is notably higher than the precision/recall for the returned class, which reflects performance at a single fixed threshold (0.5) rather than across all thresholds — a common pattern with imbalanced data.

## Handling Class Imbalance

Because `Returned` orders make up only 6.9% of the dataset, the baseline logistic regression model had limited ability to identify them (56% recall). To test whether this could be improved, the model was retrained with `class_weight='balanced'`, which increases the training penalty for misclassifying the minority class.

The computed class weights were:

| Class            | Weight |
| ---------------- | -----: |
| Not Returned (0) |   0.54 |
| Returned (1)     |   7.30 |

This means an error on a `Returned` order was penalized roughly **13.6x** more heavily than an error on a `Not Returned` order during training.

### Balanced Model Results

| Metric                       |  Score |
| ----------------------------- | -----: |
| **Accuracy**                  | 93.17% |
| **Precision (Not Returned)**  |   100% |
| **Recall (Not Returned)**     |    93% |
| **F1-Score (Not Returned)**   |    96% |
| **Precision (Returned)**      |    50% |
| **Recall (Returned)**         |   100% |
| **F1-Score (Returned)**       |    67% |
| **ROC-AUC**                   | 96.87% |

**Confusion Matrix (balanced model):**

| Actual / Predicted   | Not Returned (0) | Returned (1) |
| --------------------- | ---------------: | -----------: |
| **Not Returned (0)**  |           23,844 |        1,888 |
| **Returned (1)**      |                0 |        1,892 |

### Interpretation

`class_weight='balanced'` shifted the model's decision threshold rather than improving its underlying discriminative ability — ROC-AUC stayed essentially unchanged (96.88% → 96.87%), since AUC measures ranking quality across all thresholds, not performance at any single cutoff.

The practical effect was a tradeoff:
* **Recall for `Returned` rose from 56% to 100%** — the model now catches every actual return in the test set.
* **Precision for `Returned` fell from 55% to 50%** — roughly half of the orders it flags as "will be returned" are false alarms.

Which version is preferable depends on business cost: if failing to catch a real return is more costly than investigating a false alarm, the balanced model is the better choice. If false alarms carry a high operational cost, the original (unweighted) model's more moderate tradeoff may be preferable.

## Visualizations

*(Insert ROC curve and confusion matrix plots here.)*

* **ROC Curve** — plots the true positive rate against the false positive rate across all classification thresholds; the model's curve sits well above the random-guess diagonal.
* **Confusion Matrix** — visual breakdown of correct vs. incorrect predictions for each class.

### Interpretation

The model achieved an accuracy of **93.86%** on the test dataset.

However, because the dataset is imbalanced, accuracy alone does not fully represent the model's performance. The model achieved a **56% recall for returned orders**, meaning it correctly identified 1,059 of the 1,892 actual returned orders.

### Main Findings

* The model achieved **93.86% overall accuracy** on the test dataset.
* The model performed better at identifying **non-returned orders** than returned orders.
* The **56% recall for returned orders** indicates that the model missed a substantial share of actual returns.
* The class imbalance in the dataset likely affected the model's ability to identify returned orders.
* An early version of the model leaked the target through `order_status` and `payment_status`; identifying and removing this leakage was a key step in producing a trustworthy result.

## Limitations

* The dataset has a significant class imbalance.
* The model's performance on returned orders is lower than its performance on non-returned orders.
* The dataset may not represent all e-commerce customers and orders.
* Results may not generalize to other datasets or real-world situations.
* Additional features (e.g., product category, which was not merged into this dataset) and alternative machine learning algorithms could potentially improve performance.

## Conclusion

This project demonstrates how machine learning can be used to predict e-commerce returns. A Logistic Regression model was trained using **119 preprocessed features** and achieved **93.86% accuracy** on the test dataset.

Although the model performed well overall, its ability to identify returned orders was more limited, with a recall of **56%**. Future improvements could include addressing the class imbalance (e.g., `class_weight='balanced'` or resampling), adding relevant features such as product category, and testing other machine learning algorithms.

## Technologies

* Python
* Pandas
* Scikit-learn
* Matplotlib