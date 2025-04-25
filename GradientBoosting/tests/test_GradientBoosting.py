import os
import numpy as np
import pandas as pd
import pytest
from model.GradientBoosting import GradientBoostingClassifier
from model.GradientBoostingResults import GradientBoostingResults

# Determine the path to the datasets folder once
tests_dir = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(tests_dir, '..', 'datasets'))

def load_csv(name):
    """
    Helper to load a CSV from our datasets directory.
    """
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path)

# TEST 1: Perfect accuracy on a linearly separable dataset
def test_linearly_separable_perfect_accuracy():
    print("\n" + "="*30)
    print("TEST 1: Perfect accuracy on linearly separable data")
    print("="*30)

    # Load data and split into features/labels
    df = load_csv("Linear_dataset.csv")
    X = df.drop("y", axis=1).values
    y = df["y"].values

    # Very simple stump learners should achieve near-perfect separation
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=1)
    model.fit(X, y)

    # Wrap into our results helper for metrics & prints
    results = GradientBoostingResults(model, X, y)
    preds = results.y_pred

    # Show a few predictions vs. actual
    print("Predictions (first 10):", preds[:10])
    print("Actual      (first 10):", y[:10])

    # Print accuracy, precision, recall, F1, log loss
    results.summary()

    # Assert we have extremely high accuracy
    acc = results.accuracy()
    assert acc > 0.95, f"Expected >95% accuracy on perfectly separable data, got {acc:.2f}"

# TEST 2: Handle noise but still do reasonably well
def test_noisy_linear_data():
    print("\n" + "="*30)
    print("TEST 2: Good performance on noisy linear data")
    print("="*30)

    # Load our noisy linear dataset
    df = load_csv("noisy_linear_dataset.csv")
    X = df.drop("y", axis=1).values
    y = df["y"].values

    # Grow a slightly deeper model to accommodate noise
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    results = GradientBoostingResults(model, X, y)
    preds = results.y_pred

    # Quick peek at predictions vs. labels
    print("Predictions (first 10):", preds[:10])
    print("Actual      (first 10):", y[:10])

    # Print all classification metrics
    results.summary()

    # We expect decent accuracy despite noise
    acc = results.accuracy()
    assert acc > 0.80, f"Expected >80% accuracy on noisy data, got {acc:.2f}"

# TEST 3: High-dimensional but only a few features matter
def test_high_dimensional_sparse():
    print("\n" + "="*30)
    print("TEST 3: High-dimensional sparse dataset")
    print("="*30)

    # Load synthetic high-dim sparse data
    df = load_csv("high_dim_sparse_dataset.csv")
    X = df.drop("y", axis=1).values
    y = df["y"].values

    # Use decision stumps as weak learners
    clf = GradientBoostingClassifier(
        n_estimators=20,     # few rounds
        max_depth=1,         # stump depth
        base_learner="stump" # force stump usage
    )
    clf.fit(X, y)
    results = GradientBoostingResults(clf, X, y)
    preds = results.y_pred

    # Inspect a handful of predictions
    print("Predictions (first 10):", preds[:10])
    print("Actual      (first 10):", y[:10])

    results.summary()

    # Should pick up the few true features and classify reasonably
    acc = results.accuracy()
    assert acc > 0.75, f"Expected >75% accuracy on sparse high-dim data, got {acc:.2f}"

# TEST 4: Imbalanced classes—prioritize precision and recall
def test_imbalanced_classes_precision_recall():
    print("\n" + "="*30)
    print("TEST 4: Imbalanced classes handling")
    print("="*30)

    # Load a dataset with far fewer positives than negatives
    df = load_csv("imbalance_dataset.csv")
    X = df.drop("y", axis=1).values
    y = df["y"].values

    # A deeper tree helps capture minority signals
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    results = GradientBoostingResults(model, X, y)
    preds = results.y_pred

    print("Predictions (first 10):", preds[:10])
    print("Actual      (first 10):", y[:10])
    results.summary()

    # Check that we aren't completely ignoring the rare class
    prec = results.precision()
    rec  = results.recall()
    assert prec > 0.5, f"Precision should be >0.5 on imbalanced data, got {prec:.2f}"
    assert rec  > 0.3, f"Recall should be >0.3 on imbalanced data, got {rec:.2f}"

# TEST 5: Plotting the log-loss curve to disk
def test_plot_loss_curve(tmp_path):
    print("\n" + "="*30)
    print("TEST 5: plot_loss_curve saves a PNG")
    print("="*30)

    df = load_csv("noisy_linear_dataset.csv")
    X = df.drop("y", axis=1).values
    y = df["y"].values

    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.2, max_depth=2)

    # Before fitting, should raise because no loss has been recorded yet
    with pytest.raises(ValueError):
        model.plot_loss_curve("will_not_exist.png")

    # Fit and then save
    model.fit(X, y)
    out = tmp_path / "loss.png"
    model.plot_loss_curve(str(out))

    # Ensure the file was really created
    assert out.exists() and out.stat().st_size > 0
    print(f">>> Loss curve written to: {out}")

# TEST 6: Edge-case where all labels are identical
def test_all_same_label():
    print("\n" + "="*30)
    print("TEST 6: All labels same edge case")
    print("="*30)

    df = load_csv("Linear_dataset.csv")
    X = df.drop("y", axis=1).values
    y = np.zeros_like(df["y"].values)  # force all zeros

    model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    results = GradientBoostingResults(model, X, y)
    preds = results.y_pred

    # Model can still run; accuracy is trivial here
    print("Predictions (first 10):", preds[:10])
    print("Actual      (first 10):", y[:10])
    results.summary()

    # Accuracy should be perfect since all labels match
    assert results.accuracy() == 1.0

# TEST 7: predict_proba outputs values in [0,1]
def test_predict_proba_range():
    print("\n" + "="*30)
    print("TEST 7: predict_proba returns valid probability range")
    print("="*30)

    df = load_csv("Linear_dataset.csv")
    X = df[['x_1','x_2']].values
    y = df['y'].values

    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(X, y)
    probs = model.predict_proba(X)

    print("Probabilities (first 10):", probs[:10])
    print(f">>> min(proba) = {probs.min():.4f}, max(proba) = {probs.max():.4f}")

    # Ensure all probabilities are valid
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
