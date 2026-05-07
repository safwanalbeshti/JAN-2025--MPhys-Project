from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate linearly separable data
X, y = make_classification(
    n_samples=500,  # Number of data points
    n_features=2,   # Number of features (dimensions)
    n_informative=2,  # Number of informative features (relevant for classification)
    n_redundant=0,  # No redundant features
    n_clusters_per_class=1,  # Each class forms a single cluster
    class_sep=2.0,  # Separation between classes
    random_state=42  # For reproducibility
)

# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')
plt.title("Linearly Separable Data", fontsize=16)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
