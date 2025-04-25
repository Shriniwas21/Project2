import os
import numpy as np
from model.GradientBoosting import GradientBoostingClassifier

def test_gradient_boosting_with_subsampling():
    X = np.array([
        [1], [2], [3], [4], [5],
        [10], [11], [12], [13], [14]
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, subsample=0.8)
    model.fit(X, y)

    assert len(model.loss_list) == 50, "Subsampled model didn't run for all boosting rounds"

    plot_path = "subsampling_loss_curve.png"
    model.plot_loss_curve(plot_path)
    full_path = os.path.join("plots", plot_path)

    assert os.path.exists(full_path), f"Loss plot not saved: {full_path}"

    print("Test passed: Subsampling works and plot saved at", full_path)
