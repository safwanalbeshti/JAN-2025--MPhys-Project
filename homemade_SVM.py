import numpy as np


X_train_1 = np.array([np.random.rand(5) for x in range(5)])
y_train_1 = np.array([np.dot(x, x) for x in X_train_1])
y_train_1 = np.where(y_train_1 > 1.5, 1, -1)
print(X_train_1, y_train_1)

class homemade_SVM():
    def __init__(self, kernel):
        self.kernel = kernel
    def hinge_loss(self, predictions, y_train):
        products = y_train * predictions
        maxes = np.maximum(products, np.zeros(products.shape[0]))
        return np.sum(maxes)
    def fit(self, X_train, y_train):
        X_T_dot_X = np.multiply(X_train.T, X_train)
        X_T_dot_X += 1
        d = 2
        X_T_dot_X = X_T_dot_X ** d
        alpha, b = np.random.rand(X_T_dot_X.shape[0]), 0
        X_T_dot_X_times_alpha = np.multiply(X_T_dot_X, alpha)
        predictions = np.sum(X_T_dot_X_times_alpha, axis=-1) + b
        hinge_loss = self.hinge_loss(predictions, y_train)
        return hinge_loss
x = homemade_SVM("keren")
x.fit(X_train_1, y_train_1)