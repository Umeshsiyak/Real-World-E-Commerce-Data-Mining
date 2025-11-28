**Real-World-E-Commerce-Data-Mining**

This project analyzes the Olist Brazilian e-commerce dataset to uncover delivery performance drivers, build customer segments with K-Means, and forecast sales using SARIMA and Prophet. Insights help optimize logistics, marketing strategy, and demand planning.
Brazilian E-Commerce Analysis: Delivery Performance, Customer Segmentation & Sales Forecasting

**Overview:**
This project analyzes the Brazilian Olist E-commerce Public Dataset to understand key operational, customer, and sales patterns. Using machine learning and statistical modeling, the project answers three core questions:

1. What factors influence delivery performance?

2. How can customers be segmented using behavioral features?

3. Can we reliably forecast future sales volumes?

**Dataset:**
We use the Olist Brazilian E-commerce Public Dataset, which includes:

Orders

Order items

Customers

Payments

Products

Sellers

Reviews

Geolocation

Over 100k orders (2016–2018) were merged and engineered into analysis-ready datasets.

**Project Components:**
**1️. Delivery Performance Analysis**

Engineered delivery delay, delivery time, shipping distance, and product-level features.

Compared Linear Regression, Random Forest, and Gradient Boosting.

Key Insight: Shipping distance is the strongest predictor of delivery delays.

Best Model: Gradient Boosting (R² = 0.64).

2️. Customer Segmentation (K-Means)

Built Recency, Frequency, Monetary (RFM) features.

Added behavioral indicators (category diversity, repeat status).

Identified 5 segments, including:

Dormant mid-value customers

Recent high-value explorers

Recent low-value buyers

Insight: Overall repeat-purchase rates are extremely low.

3️. Time Series Forecasting

Aggregated daily sales from delivered orders.

Modelled using:

SARIMA

Seasonal Naive Benchmark

Prophet

Weekly seasonality and late-2017 demand spikes observed.

Seasonal naive unexpectedly outperformed advanced models on MAPE and RMSE.

**Key Takeaways:**

Delivery speed depends primarily on distance, product size, and location.

Customer base shows low retention, requiring targeted retention strategies.

Sales forecasting captures trend/seasonality but needs additional regressors to outperform naïve models.

**Technologies Used:**

Python (Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Prophet)

Jupyter Notebook

GitHub for version control

📎 References

Kaggle Olist Dataset, Hyndman & Athanasopoulos (Forecasting Principles), Prophet documentation, and supporting statistical modeling literature.
