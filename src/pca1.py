import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = 'data/processed_nasa_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Apply PCA without specifying the number of components to analyze the explained variance
pca = PCA()
principal_components = pca.fit_transform(data.drop(columns=['Hazardous']))  # Assuming 'Hazardous' is the target column

# Explained variance by each principal component
explained_variance = pca.explained_variance_ratio_
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
data_numeric = data.drop(non_numeric_columns, axis=1)
data_hazardous = data['Hazardous']
# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Applying PCA
pca = PCA(n_components=4)
principal_components = pca.fit_transform(data_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4'])

# Variance explained by each principal component
explained_variance = pca.explained_variance_ratio_
pca_df['Hazardous'] = data_hazardous
# Plotting various charts
sns.pairplot(pca_df,hue='Hazardous')
plt.show()
# 1. Scree plot of explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='g', label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('Scree Plot')
plt.show()

# 2. Cumulative explained variance plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o', linestyle='--', color='b')
plt.ylabel('Cumulative explained variance')
plt.xlabel('Principal components')
plt.title('Cumulative Explained Variance Plot')
plt.show()

# 3. Pair plot of the first two principal components
sns.pairplot(pca_df[['PC1', 'PC2']])
plt.show()

# 4. Heatmap of the principal components correlation with features
plt.figure(figsize=(12, 6))
sns.heatmap(pd.DataFrame(pca.components_, columns=data_numeric.columns, index=['PC1', 'PC2', 'PC3', 'PC4']), cmap='viridis')
plt.title('PCA Component Heatmap')
plt.show()

# Creating a table of principal component values
pca_values_table = pd.DataFrame(pca.components_, columns=data_numeric.columns, index=['PC1', 'PC2', 'PC3', 'PC4'])
pca_values_table.head()
# Plotting the explained variance for each principal component
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.title('Explained Variance by Different Principal Components')
plt.show()

# Optional: Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: First two principal components')
plt.grid(True)
plt.show()
