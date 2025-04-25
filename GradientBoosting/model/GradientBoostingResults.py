import numpy as np

class GradientBoostingResults:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        self.y_proba = self.model.predict_proba(X_test)

    def accuracy(self):
        return np.mean(self.y_pred == self.y_test)

    def precision(self):
        tp = np.sum((self.y_pred == 1) & (self.y_test == 1))
        fp = np.sum((self.y_pred == 1) & (self.y_test == 0))
        return tp / (tp + fp + 1e-10)

    def recall(self):
        tp = np.sum((self.y_pred == 1) & (self.y_test == 1))
        fn = np.sum((self.y_pred == 0) & (self.y_test == 1))
        return tp / (tp + fn + 1e-10)

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r + 1e-10)

    def log_loss(self):
        eps = 1e-10
        p = np.clip(self.y_proba, eps, 1 - eps)
        return -np.mean(self.y_test * np.log(p) + (1 - self.y_test) * np.log(1 - p))

    def summary(self):
        print("----------------------")
        print(f"Accuracy : {self.accuracy():.4f}")
        print(f"Precision: {self.precision():.4f}")
        print(f"Recall   : {self.recall():.4f}")
        print(f"F1 Score : {self.f1_score():.4f}")
        print(f"Log Loss : {self.log_loss():.4f}")
