# importing libraries

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from collections import Counter

# Loading dataset
digits = load_digits()

# Dataset split
def train_test_split(X, y, test_size_per_class):
    classes = np.unique(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    for c in classes:
        idx = np.where(y == c)
        X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X[idx], y[idx], test_size=test_size_per_class)
        X_train.extend(X_c_train)
        X_test.extend(X_c_test)
        y_train.extend(y_c_train)
        y_test.extend(y_c_test)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size_per_class=50)

# Nearest Neighbor
def nn(train_data, train_labels, test_sample):
    dist = np.linalg.norm(train_data - test_sample, axis=1)
    nearest_idx = np.argmin(dist)
    return train_labels[nearest_idx]

correct_pred = 0
for i, test_sample in enumerate(X_test):
    pred_label = nn(X_train, y_train, test_sample)
    if pred_label == y_test[i]:
        correct_pred += 1
accuracy_nn = correct_pred / len(X_test) * 100
print(f"Nearest Neighbor Classification Accuracy: {accuracy_nn:.2f}%")

# k-Nearest Neighbors
def knn(train_data, train_labels, test_sample, k):
    dist = np.linalg.norm(train_data - test_sample, axis=1)
    k_nearest_indices = np.argsort(dist)[:k]
    k_nearest_labels = train_labels[k_nearest_indices]
    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_label

for k in [3, 5, 7]:
    correct_pred = 0
    for i, test_sample in enumerate(X_test):
        pred_label = knn(X_train, y_train, test_sample, k)
        if pred_label == y_test[i]:
            correct_pred += 1
    accuracy_knn = correct_pred / len(X_test) * 100
    print(f"{k}-Nearest Neighbors Classification Accuracy: {accuracy_knn:.2f}%")
