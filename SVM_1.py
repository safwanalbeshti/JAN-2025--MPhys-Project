import numpy as np
from cvxopt import matrix, solvers
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate the dataset
X, y = make_classification(
    n_samples=1000,  # Number of data points
    n_features=2,   # Number of features (dimensions)
    n_informative=2,  # Number of informative features (relevant for classification)
    n_redundant=0,  # No redundant features
    n_clusters_per_class=1,  # Each class forms a single cluster
    class_sep=2.0,  # Separation between classes
    random_state=42  # For reproducibility
)

class SimplifiedSVM:
    def __init__(self, C=None):
        self.C = C  # Regularization parameter
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert y to float (-1, 1)
        y = y.astype(float)
        y = np.where(y == 0, -1, 1)

        # Compute the Kernel matrix (linear kernel)
        K = np.dot(X, X.T)

        # Prepare matrices for quadratic programming
        P = matrix(np.array(np.outer(y, y) * K, dtype=np.float64))
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C if self.C else np.inf)))
        A = matrix(np.array(y.reshape(1, -1), dtype=np.float64))
        b = matrix(0.0)

        # Solve the quadratic programming problem
        solvers.options["show_progress"] = False
        sol = solvers.qp(P, q, G, h, A, b)

        alphas = np.array(sol['x']).flatten()

        # Support vectors have non-zero lagrange multipliers
        sv = alphas > 1e-5
        self.alpha = alphas[sv]
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]

        # Compute weight vector (only for linear kernel)
        self.w = np.sum(self.alpha[:, None] * self.support_vector_labels[:, None] * self.support_vectors, axis=0)

        # Compute bias term
        self.b = np.mean(
            self.support_vector_labels - np.dot(self.support_vectors, self.w)
        )
        print(self.w,self.b)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Train the SVM
svm = SimplifiedSVM(C=1.0)
svm.fit(X, y)

# Predict
print("Predictions:", svm.predict(X))
print(np.sum(abs(svm.predict(X) - np.where(y==0,-1,1)))/2)

x = np.linspace(min(X[:,0]), max(X[:,0]), 400)
def y(x):
    y = (-1 * svm.w[0] * x - svm.b) / svm.w[1]
    return y

colors = ["red" if svm.predict(x_i) == 1 else "blue" for x_i in X]
print(len(colors))
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color=colors, s=80, edgecolors='k')
plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], s=100, facecolors='pink', edgecolors='k', label='Support Vectors')
plt.plot(x, y(x))
plt.grid(True)
plt.show()
