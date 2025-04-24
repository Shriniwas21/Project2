import numpy as np

class DecisionStump:

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue 

                left_value = np.mean(y[left_mask])
                right_value = np.mean(y[right_mask])

                predictions = np.where(left_mask, left_value, right_value)
                error = np.mean((y - predictions) ** 2)

                if error < min_error:
                    min_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        return np.where(feature_values <= self.threshold, self.left_value, self.right_value)



class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.initial_prediction = 0.0  # usually log(odds)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass
