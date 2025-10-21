import numpy as np


class NaiveBayesClassifier:
    """Naïve Bayes classifier with binary features"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.class_priors = None
        self.feature_probs = None
        self.classes = None

    def fit(self, X_train, y_train):
        """Train Naïve Bayes classifier"""
        # Binarize images
        X_binary = (X_train.reshape(X_train.shape[0], -1) > self.threshold).astype(int)

        self.classes = np.unique(y_train)
        n_features = X_binary.shape[1]

        # Initialize probability arrays
        self.class_priors = np.zeros(len(self.classes))
        self.feature_probs = np.zeros((len(self.classes), n_features))

        # Calculate priors and conditional probabilities
        for idx, c in enumerate(self.classes):
            X_c = X_binary[y_train == c]
            self.class_priors[idx] = len(X_c) / len(X_binary)

            # P(feature=1|class) with Laplace smoothing
            self.feature_probs[idx] = (np.sum(X_c, axis=0) + 1) / (len(X_c) + 2)

        print(f"Naïve Bayes trained on {len(X_binary)} samples")

    def predict(self, X_test):
        """Predict using Bayes' rule"""
        X_binary = (X_test.reshape(X_test.shape[0], -1) > self.threshold).astype(int)
        predictions = []

        for test_sample in X_binary:
            log_probs = np.log(self.class_priors)

            for idx in range(len(self.classes)):
                # Log probability to avoid underflow
                log_probs[idx] += np.sum(
                    test_sample * np.log(self.feature_probs[idx] + 1e-10) +
                    (1 - test_sample) * np.log(1 - self.feature_probs[idx] + 1e-10)
                )

            predictions.append(self.classes[np.argmax(log_probs)])

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        """Compute accuracy"""
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy, predictions


