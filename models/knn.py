import numpy as np
from collections import Counter


class KNNClassifier:
    """K-Nearest Neighbor classifier using only NumPy"""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Store training data"""
        # Flatten images to vectors
        self.X_train = X_train.reshape(X_train.shape[0], -1)
        self.y_train = y_train
        print(f"KNN trained with {len(self.X_train)} samples")

    def predict(self, X_test):
        """Predict labels for test data"""
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        predictions = []

        for i, test_sample in enumerate(X_test_flat):
            if i % 100 == 0:
                print(f"Predicting sample {i}/{len(X_test_flat)}")

            # Compute Euclidean distances to all training samples
            distances = np.sqrt(np.sum((self.X_train - test_sample) ** 2, axis=1))

            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]

            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        """Compute accuracy"""
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy, predictions


