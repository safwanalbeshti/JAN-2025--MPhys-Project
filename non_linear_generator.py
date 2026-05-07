# import numpy as np
#
# X = np.array([[2,3]])
# def non_linear_generator(X, rounds):
#     X = np.hstack([np.ones((X.shape[0], 1)), X])
#     outer_products = np.einsum('ij,ik->ijk', X, X)  # Shape: (n_samples, n_features, n_features)
#     tril_indices = np.tril_indices(X.shape[1])
#     tril_values = outer_products[:, tril_indices[0], tril_indices[1]]  # Shape: (n_samples, num_tril_elements)
#     if rounds > 1:
#         return non_linear_generator(tril_values, rounds-1)
#     else:
#         return tril_values
#
# print(non_linear_generator(X,1))
# print(non_linear_generator(X,2))
#
# import random
# subset = random.choices(range(1, 21), k=5)
# print(subset)