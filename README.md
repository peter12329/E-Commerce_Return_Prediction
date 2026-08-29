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

The following columns were excluded because they were identifiers, post-return information, or considered irrelevant to prediction:

* `order_id`
* `customer_id`
* `customer_name`
* `order_date`
* `order_time`
* `return_status`
* `return_reason`
* `customer_review`
* `review_sentiment`
* `customer_rating`
* `campaign_name`
* `coupon_code`
* `customer_postal_code`
* `customer_city`
* `order_status`
* `payment_status`

### Target Variable

The target variable `is_returned` was created from `return_status`.

| Value | Meaning      |
| ----- | ------------ |
| 0     | Not Returned |
| 1     | Returned     |

## Initial Data Findings

Before training the model, the following observations were found:

* The dataset contains **138,116 records**.
* **128,654 (93.15%)** orders were not returned.
* **9,462 (6.85%)** orders were returned.
* The target variable `is_returned` is **imbalanced**, with significantly more non-returned orders than returned orders.

### Missing Values

Missing values in `delivery_days` and `estimated_delivery_days` were handled using the median value of each respective column. After preprocessing, both columns contained **0 missing values**.

### Target Distribution

| Target           |   Count | Percentage |
| ---------------- | ------: | ---------: |
| Not Returned (0) | 128,654 |     93.15% |
| Returned (1)     |   9,462 |      6.85% |

## Data Preprocessing

1. Created the binary target variable `is_returned`.
2. Removed identifier, post-return, and potentially irrelevant columns.
3. Filled missing values in `delivery_days` and `estimated_delivery_days` using the median.
4. Converted categorical variables using one-hot encoding.
5. Split the data into **80% training** and **20% testing** data.
6. Standardized the features using `StandardScaler`.

After one-hot encoding, the dataset contained **119 model features**.

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
| ---------------------------- | -----: |
| **Accuracy**                 | 93.86% |
| **Precision (Not Returned)** |    97% |
| **Recall (Not Returned)**    |    97% |
| **F1-Score (Not Returned)**  |    97% |
| **Precision (Returned)**     |    55% |
| **Recall (Returned)**        |    56% |
| **F1-Score (Returned)**      |    56% |

### Confusion Matrix

| Actual / Predicted   | Not Returned (0) | Returned (1) |
| -------------------- | ---------------: | -----------: |
| **Not Returned (0)** |           24,869 |          863 |
| **Returned (1)**     |              833 |        1,059 |

### Interpretation

The model achieved an accuracy of **93.86%** on the test dataset.

However, because the dataset is imbalanced, accuracy alone does not fully represent the model's performance. The model achieved a **56% recall for returned orders**, meaning it correctly identified 1,059 of the 1,892 actual returned orders.

### Main Findings

* The model achieved **93.86% overall accuracy** on the test dataset.
* The model performed better at identifying **non-returned orders** than returned orders.
* The **56% recall for returned orders** indicates that the model missed some actual returns.
* The class imbalance in the dataset may have affected the model's ability to identify returned orders.

## Limitations

* The dataset has a significant class imbalance.
* The model's performance on returned orders is lower than its performance on non-returned orders.
* The dataset may not represent all e-commerce customers and orders.
* Results may not generalize to other datasets or real-world situations.
* Additional features and alternative machine learning algorithms could potentially improve performance.

## Conclusion

This project demonstrates how machine learning can be used to predict e-commerce returns. A Logistic Regression model was trained using **119 preprocessed features** and achieved **93.86% accuracy** on the test dataset.

Although the model performed well overall, its ability to identify returned orders was more limited, with a recall of **56%**. Future improvements could include addressing the class imbalance, adding relevant features, and testing other machine learning algorithms.

## Technologies

* Python
* Pandas
* Scikit-learn
