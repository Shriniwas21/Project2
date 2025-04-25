import numpy as np
from model.GradientBoosting import GradientBoostingClassifier

def test_gradient_boosting_on_simple_data():
    X = np.array([
        [1], [2], [3], [4], [5],
        [10], [11], [12], [13], [14]
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    model = GradientBoostingClassifier(n_estimators=5, learning_rate=0.5)
    model.fit(X, y)
    model.plot_loss_curve("testcase1_loss_plot.png")
    predictions = model.predict(X)

    print("Predictions:", predictions)
    print("Expected:", y)

    assert np.all(predictions == y), "Model failed on simple binary split"
