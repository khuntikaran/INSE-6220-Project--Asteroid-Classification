import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'data/processed_nasa_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Using only the first two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data.drop(columns=['Hazardous']))
y = data['Hazardous']

# Standardizing the features (important for many models)
scaler = StandardScaler()
X_pca = scaler.fit_transform(X_pca)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Training a logistic regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Creating a mesh to plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plotting decision boundary
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', s=20)
plt.title('Decision Boundary with First Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


