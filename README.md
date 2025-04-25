## Team Members
1. Nishant Uday Dalvi - ndalvi@hawk.iit.edu **(A20556507)** (Member 1)
2. Shriniwas Oza - soza1@hawk.iit.edu **(A20568892)** (Member 2)

## Project 2 - Gradient Boosting Tree Classifier

## Overview
This project implements a **Gradient Boosting Tree Classifier** from first principles using NumPy. The model supports:
- Stumps and full decision trees as weak learners
- Subsampling (stochastic boosting)
- Feature importance tracking
- Log-loss tracking and plotting
- Full evaluation metrics without scikit-learn

We followed Sections 10.9-10.10 of *Elements of Statistical Learning (2nd Edition)* and enforced a strict **no scikit-learn / no boosting libraries** rule.

---

## Model Design and Implementation
- `GradientBoostingClassifier`: The core boosting class implementing `fit()`, `predict()`, and `predict_proba()` using log-loss gradients and trees as weak learners.
- `DecisionStump`: A basic one-level decision tree for fast, interpretable updates.
- `DecisionTreeRegressor`: A custom recursive regressor supporting `max_depth`, used as the base learner.
- `GradientBoostingResults`: Outputs accuracy, precision, recall, F1, and log-loss - no sklearn used.
- `plot_loss_curve()`: Saves training loss (and validation loss if given) as PNGs for reproducibility.

---

## Test Design and Implementation

- The test suite is built using `pytest` and includes **both unit and integration-level tests**. It ensures:
  - Correctness on toy datasets
  - Generalization on noisy and sparse data
  - Stability on imbalanced class distributions
  - Validity of output probabilities (`predict_proba`)
  - File output correctness for plots

- Evaluation Metrics:
    - `GradientBoostingResults` evaluates predictions using:
      - Accuracy
      - Precision
      - Recall
      - F1 Score
      - Log-loss
    - These metrics are implemented **without using scikit-learn**

- Edge Case Handling:
    - All labels same
    - High-dimensional features with low signal
    - Data with high class imbalance
    - Probabilistic output range and plotting

- Model Validation Enhancements:
    - Tests check `plot_loss_curve()` generates PNGs
    - Metric summaries printed for human-readable traceability
    - Threshold-based assertions on accuracy, precision, and recall included

---

## Folder Structure
```
Project2/
├── GradientBoosting/
│   ├── model/
│   │   ├── GradientBoosting.py
│   │   └── GradientBoostingResults.py
│   ├── tests/
│   │   ├── plots/
│   │   ├── images/
│   │   ├── testcase1_loss_plot.png
│   │   ├── subsampling_loss_curve.png
│   │   ├── test_DecisionStump.py
│   │   ├── test_GradientBoosting.py
│   │   ├── test_GradientBoostingResults.py
│   │   ├── test_subsampling.py
│   │   └── test_feature_importance.py
│   ├── datasets/
│   │   ├── Linear_dataset.csv
│   │   ├── generate_classification_data.py
│   │   ├── high_dim_sparse_dataset.csv
│   │   ├── imbalance_dataset.csv
│   │   └── noisy_linear_dataset.csv
│   ├── plots/
│   │   ├── testcase1_loss_plot.png
│   │   ├── subsampling_loss_curve.png
│   │   ├── long_Loss_curve_over_boosting.png
├── README.md
├── requirements.txt

```

## How to Run

### Step 1: Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run the tests using PyTest
```bash
pytest -s GradientBoosting/tests/
```
The `-s` flag prints formatted output for each test.

You will see **10 test outputs**, each formatted with headers and evaluation metrics.

### Example Usage
```python
from GradientBoosting.model.GradientBoosting import GradientBoostingClassifier
from GradientBoosting.model.GradientBoostingResults import GradientBoostingResults
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [10], [11], [12]])
y = np.array([0, 0, 0, 1, 1, 1])

# Fit model with trees
model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, base_learner="tree", subsample=0.8)
model.fit(X, y)
model.plot_loss_curve("plots/demo_loss.png")

# Evaluate
results = GradientBoostingResults(model, X, y)
results.summary()

# Feature importances
print(model.feature_importances_(n_features=X.shape[1]))
```

## What the Model Does
This is a Gradient Boosting Tree Classifier for binary classification, trained using:
- Log-loss gradients
- Decision trees or stumps as weak learners
- Additive updates to minimize loss

It's ideal when:
- You want an ensemble that builds sequentially and corrects mistakes iteratively
- You have structured tabular data
- You need model flexibility and interpretability (via feature importance and visualization)

## How We Tested the Model

We implemented multiple **unit tests** using PyTest, with real-world inspired datasets. Each test is structured to check a specific scenario:

- `test_DecisionStump.py`: Validates simple stump prediction on toy data
- `test_feature_importance.py`: Ensures that feature importances are tracked and normalized
- `test_subsampling.py`: Verifies log-loss convergence and saved plots with sampled data
- `test_GradientBoostingResults.py`: Tests metric calculations (accuracy, precision, recall, F1, log-loss)
- `test_GradientBoosting.py`: Comprehensive coverage on:
  - Perfectly separable data
  - Noisy linear regression data
  - High-dimensional sparse datasets
  - Imbalanced classes (precision/recall check)
  - Constant-label edge cases
  - Output of `predict_proba` in [0, 1]
  - Plot saving using `plot_loss_curve()`

All test files use PyTest conventions and datasets are loaded from the `datasets/` directory for reproducibility and grading transparency.


**Test Outputs:**
Test output screenshots are saved in `./GradientBoosting/tests/images/` as `test_*.png`.

## Limitations / Known Challenges
- Unstable predictions on small noisy datasets when using deep trees and high learning rates.
- Constant target labels can lead to near-zero gradients and stagnation.
- Highly correlated features may lead to split selection instability (expected).

## Answering the README Questions

### What does the model do and when should it be used?
This project implements a **Gradient Boosting Tree Classifier** from scratch using additive stage-wise learning. The model uses decision trees or stumps as base learners to iteratively reduce prediction error using the gradient of the log-loss function.

This classifier is particularly effective for:
- Structured/tabular binary classification tasks
- Datasets where **boosting with shallow trees** can outperform deep individual models
- Tasks requiring **feature importance**, **flexible model control**, or **log-loss optimization**

### How did you test your model?
We developed a comprehensive test suite to validate model behavior across components:
- Verified correctness with clearly separable datasets
- Tested both **stump** and **tree** base learners
- Validated log-loss reduction over time with saved plots
- Included tests for **feature importance**, **subsampling**, and full evaluation metrics
- Used `GradientBoostingResults` to report **accuracy, precision, recall, F1, and log-loss** — fully custom

All tests run via `pytest` and are reproducible.

### What parameters have you exposed to users?
```python
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    base_learner="tree",
    subsample=0.8,
    early_stopping_rounds=None
)
```
- `n_estimators`: Number of boosting iterations
- `learning_rate`: Controls contribution of each learner
- `max_depth`: Controls depth of decision trees
- `base_learner`: Choose between "tree" and "stump"
- `subsample`: Enables stochastic boosting by sampling training data

### Are there inputs the model struggles with? Given more time, could you work around these or is it fundamental?
The model can struggle with:
- Very small datasets combined with high n_estimators, causing overfitting
- Constant features (very low variance), which contribute little to splitting
- Highly imbalanced data, where gradient updates may bias early predictions

## Final Notes
This submission includes:
- This project strictly avoids any use of scikit-learn or external boosting libraries. All components were implemented from scratch using NumPy.
- We implemented **advanced features** like subsampling, configurable learners, log-loss visualization, and feature importances to mirror real-world boosting libraries (e.g., XGBoost, LightGBM).

