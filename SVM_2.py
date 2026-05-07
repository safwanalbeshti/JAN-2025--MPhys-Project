import numpy as np
from cvxopt import matrix, solvers
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from non_linear_generator import non_linear_generator

# Generate the dataset
X, y = make_moons(
    n_samples=100,
    noise=0,
    random_state=42  # For reproducibility
)

# X, y = make_classification(
#     n_samples=100,
#     n_features=2,
#     n_informative=2,
#     n_redundant=0,
#     n_classes=2,
#     n_clusters_per_class=1,
#     random_state=44
# )

X_non_linearised = non_linear_generator(X,2)

class SimplifiedSVM:
    def __init__(self, C=None, kernel="linear", gamma=1.0, coef0=0.3, degree=4):
        self.C = C  # Regularization parameter
        self.w = None
        self.b = None
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert y to float (-1, 1)
        y = y.astype(float)
        y = np.where(y == 0, -1, 1)

        # Compute the Kernel matrix (linear kernel)
        if self.kernel == "linear":
            K = np.dot(X, X.T)
        elif self.kernel == "poly":
            K = (self.gamma * np.dot(X, X.T) + self.coef0) ** self.degree

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
        self.sv = sv
        self.alpha = alphas
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]

        # Compute weight vector (only for linear kernel)
        if self.kernel == "linear":
            self.w = np.sum(self.alpha[:, None] * y[:,None] * X, axis=0)
            self.b = np.mean(y - np.dot(X, self.w))
        elif self.kernel == "poly":
            K = (self.gamma * np.dot(X, self.support_vectors.T) + self.coef0) ** self.degree
            self.b = np.mean(y[self.sv] - np.dot(K, self.support_vector_labels * self.alpha[self.sv])[self.sv])

    def predict(self, X):
        if self.kernel == "poly":
            K = (self.gamma * np.dot(X, self.support_vectors.T) + self.coef0) ** self.degree
            return np.sign(np.dot(K, self.support_vector_labels * self.alpha[self.sv]) + self.b)
        if self.kernel == "linear":
            return np.sign(np.dot(X, self.w) + self.b)

# Train the SVM
svm = SimplifiedSVM(C=1.0,kernel="linear", coef0=1, degree=4, gamma=1.0)
svm.fit(X_non_linearised, y)

# Predict
X_test = X
print(X_test)
X_test_non_linearised = X_non_linearised
y_test = y
predictions = svm.predict(X_test_non_linearised)
accuracy = int(np.sum(abs(predictions - np.where(y_test==0,-1,1)))/2/X_test.shape[0])
print(f"accuracy = {100 - accuracy}%")

if svm.kernel == "linear":
    x = np.linspace(min(X_test[:,0]), max(X_test[:,0]), 400)
    def y(x):
        y = (-1 * svm.w[0] * x - svm.b) / svm.w[1]
        return y
    colors = ["red" if svm.predict(x_i) == 1 else "blue" for x_i in X_test_non_linearised]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], color=colors, s=80, edgecolors='k')
    # plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], s=100, facecolors='pink', edgecolors='k', label='Support Vectors')
    # plt.plot(x, y(x))
    plt.grid(True)
    plt.show()
elif svm.kernel == "poly":
    colors_predicted = ["red" if svm.predict(x_i) == 1 else "blue" for x_i in X]
    colors_true = ["red" if y[x_i] == 1 else "blue" for x_i in range(len(X))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].scatter(X[:, 0], X[:, 1], color=colors_true, s=80, edgecolors='k')
    axes[0].set_title("True Values")
    axes[1].scatter(X[:, 0], X[:, 1], color=colors_predicted, s=80, edgecolors='k')
    # axes[1].scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], s=100, facecolors='pink', edgecolors='k', label='Support Vectors')
    axes[1].set_title("Predictions")
    plt.grid(True)
    plt.show()
