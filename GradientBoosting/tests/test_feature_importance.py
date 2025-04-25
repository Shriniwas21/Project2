import numpy as np
from model.GradientBoosting import GradientBoostingClassifier

def test_feature_importance_sum_to_1():
    X = np.array([[1, 10], [2, 20], [3, 30], [10, 100], [11, 110], [12, 120]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.2)
    model.fit(X, y)

    importances = model.feature_importances_(n_features=2)
    print("Feature importances:", importances)

    assert np.isclose(np.sum(importances), 1.0), "Importances should sum to 1"
