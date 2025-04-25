import numpy as np
import matplotlib.pyplot as plt
import os

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

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return TreeNode(value=np.mean(y))

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def _find_best_split(self, X, y):
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                mse = self._calculate_mse(y_left, y_right)
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_mse(self, y_left, y_right):
        total = len(y_left) + len(y_right)
        left_mse = np.var(y_left) * len(y_left)
        right_mse = np.var(y_right) * len(y_right)
        return (left_mse + right_mse) / total

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, base_learner="tree", subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.base_learner = base_learner
        self.subsample = subsample
        self.trees = []
        self.initial_prediction = 0.0
        self.loss_list = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        pos_ratio = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.initial_prediction = np.log(pos_ratio / (1 - pos_ratio))
        F_m = np.full(n_samples, self.initial_prediction)

        for m in range(self.n_estimators):
            prob = self._sigmoid(F_m)
            log_loss = -np.mean(y * np.log(np.clip(prob, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - prob, 1e-10, 1)))
            self.loss_list.append(log_loss)

            residuals = y - prob
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
                residuals_sample = residuals[indices]
            else:
                X_sample = X
                residuals_sample = residuals

            if self.base_learner == "stump":
                learner = DecisionStump()
            else:
                learner = DecisionTreeRegressor(max_depth=self.max_depth)

            learner.fit(X_sample, residuals_sample)
            update = learner.predict(X)

            F_m += self.learning_rate * update
            self.trees.append(learner)

    def predict_proba(self, X):
        F_m = np.full(X.shape[0], self.initial_prediction)
        for stump in self.trees:
            F_m += self.learning_rate * stump.predict(X)
        return self._sigmoid(F_m)

    def predict(self, X):
        prob = self.predict_proba(X)
        return (prob >= 0.5).astype(int)
    
    def plot_loss_curve(self, save_path="loss_plot.png"):
        if not self.loss_list:
            raise ValueError("Loss list is empty. Train the model first.")
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.loss_list) + 1), self.loss_list, marker='o', linewidth=2)
        plt.title("Log-Loss Curve over Boosting Rounds")
        plt.xlabel("Iteration")
        plt.ylabel("Log Loss")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("plots", exist_ok=True)
        full_path = os.path.join("plots", save_path)
        plt.savefig(full_path)
        print(f"[INFO] Loss curve saved to: {full_path}")
        plt.close()

    def feature_importances_(self, n_features):
        importances = np.zeros(n_features)

        for tree in self.trees:
            if hasattr(tree, 'root'):
                self._accumulate_importances(tree.root, importances)
            elif hasattr(tree, 'feature_index'):
                importances[tree.feature_index] += 1

        if np.sum(importances) > 0:
            importances /= np.sum(importances)

        return importances

    def _accumulate_importances(self, node, importances):
        if node is None or node.value is not None:
            return
        importances[node.feature_index] += 1
        self._accumulate_importances(node.left, importances)
        self._accumulate_importances(node.right, importances)

