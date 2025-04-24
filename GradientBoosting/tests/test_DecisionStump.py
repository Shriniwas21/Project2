import numpy as np
from model.GradientBoosting import DecisionStump

def test_decision_stump_fit_predict():
    X = np.array([
        [1],
        [2],
        [3],
        [10],
        [12],
        [13]
    ])
    y = np.array([1, 1, 1, -1, -1, -1])

    stump = DecisionStump()
    stump.fit(X, y)

    predictions = stump.predict(X)
    print("Feature Index:", stump.feature_index)
    print("Threshold:", stump.threshold)
    print("Left Value:", stump.left_value)
    print("Right Value:", stump.right_value)
    print("Predictions:", predictions)

    assert np.allclose(predictions, np.array([1, 1, 1, -1, -1, -1]), atol=0.5), "Test failed"
