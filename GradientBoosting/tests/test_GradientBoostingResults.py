import numpy as np
from model.GradientBoosting import GradientBoostingClassifier
from model.GradientBoostingResults import GradientBoostingResults

def test_results_metrics_are_reasonable():
    X = np.array([
        [1], [2], [3], [4], [5],
        [10], [11], [12], [13], [14]
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.2)
    model.fit(X, y)

    results = GradientBoostingResults(model, X, y)
    results.summary()

    acc = results.accuracy()
    f1 = results.f1_score()
    loss = results.log_loss()

    assert 0.9 <= acc <= 1.0, "Accuracy out of expected range"
    assert 0.9 <= f1 <= 1.0, "F1 score out of expected range"
    assert 0.0 <= loss <= 0.5, "Log loss too high for clean split"
