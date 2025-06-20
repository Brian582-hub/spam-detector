import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate synthetic customer data
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 60, size=200),
    'Annual_Income': np.random.randint(20000, 100000, size=200),
    'Spending_Score': np.random.randint(1, 100, size=200)
}

df = pd.DataFrame(data)

# 2. Data exploration
print(df.head())
print(df.describe())

# 3. Data preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 4. Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the dataframe
df['Cluster'] = clusters

# 5. Reduce to 2 dimensions for visualization using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
df['PC1'] = pca_components[:, 0]
df['PC2'] = pca_components[:, 1]

# 6. Visualize clusters using matplotlib and seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='viridis', s=100)
plt.title("Customer Segmentation using KMeans + PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()
