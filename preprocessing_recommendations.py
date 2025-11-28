import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the order-item level dataset
df = pd.read_csv("customer_data_final.csv")

# Aggregate to customer level: frequency, monetary, and basic location info
customer_level = df.groupby('customer_unique_id').agg({
    'order_id': 'nunique',
    'payment_value': 'sum',
    'customer_state': 'first',
    'customer_city': 'first',
    'customer_zip_code_prefix': 'first'
}).reset_index()

# Rename and compute average order value
customer_level.rename(columns={'order_id': 'frequency'}, inplace=True)
customer_level['avg_order_value'] = customer_level['payment_value'] / customer_level['frequency']

# Identify top 10 product categories and count purchases per customer
top_categories = df['product_category_name_english'].value_counts().nlargest(10).index
category_counts = pd.crosstab(df['customer_unique_id'], df['product_category_name_english'])
category_counts = category_counts[top_categories]

# Merge customer-level data with category counts
customer_segments = customer_level.merge(category_counts, on='customer_unique_id', how='left')
customer_segments.fillna(0, inplace=True)

# Add product category diversity (number of different categories purchased)
category_diversity = df.groupby('customer_unique_id')['product_category_name_english'].nunique()
customer_segments['category_diversity'] = category_diversity

# Add average items per order
items_per_order = df.groupby(['customer_unique_id', 'order_id'])['order_item_id'].count().reset_index()
avg_items = items_per_order.groupby('customer_unique_id')['order_item_id'].mean()
customer_segments['avg_items_per_order'] = avg_items

# TODO: Add recency, account age, return behavior, and seasonal patterns if date-related data becomes available

# Scale numerical features before clustering
num_cols = ['frequency', 'payment_value', 'avg_order_value', 'category_diversity',
            'avg_items_per_order'] + list(top_categories)
scaler = StandardScaler()
customer_segments[num_cols] = scaler.fit_transform(customer_segments[num_cols])

# Save the preprocessed dataset
customer_segments.to_csv("customer_segments_preprocessed.csv", index=False)
print(" Preprocessing complete! File saved as customer_segments_preprocessed.csv")
