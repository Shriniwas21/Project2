import numpy as np
import pandas as pd
import argparse
import csv
from sklearn.datasets import make_classification

def generate_noisy_classification_data(n_samples=200, seed=42, output="synthetic_classification.csv"):
    np.random.seed(seed)
    X1 = np.random.uniform(-3, 3, n_samples)
    X2 = np.random.uniform(-3, 3, n_samples)
    
    # Define nonlinear decision boundary
    decision_boundary = X1**2 + X2**2 < 4 
    y = decision_boundary.astype(int)
    df = pd.DataFrame({'x1': X1, 'x2': X2, 'y': y})
    df.to_csv(output, index=False)
    print(f"Noisy dataset generated with shape {df.shape}")

def generate_linear_separable(n_samples: int, centers: list, cov_scale: float, seed: int):
    rng = np.random.default_rng(seed)
    half = n_samples // 2

    # First class centered at centers[0]
    X0 = rng.multivariate_normal(mean=centers[0],cov=np.eye(2) * cov_scale,size=half)
    # Second class centered at centers[1]
    X1 = rng.multivariate_normal(mean=centers[1],cov=np.eye(2) * cov_scale,size=n_samples - half)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(half, dtype=int),np.ones(n_samples - half, dtype=int)])
    return X,y

def generate_hig_dim_sparse(n_samples: int,n_features: int,n_informative: int, flip_y: float ):
    print("running make_classification:")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        flip_y=flip_y,
        class_sep=1.0,
        random_state=42
    )

    print("make clasification successfull")

    print("now writing to csv:")

    header = [f"x_{i}" for i in range(args.n_features)] + ["y"]
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for xi, yi in zip(X, y):
            writer.writerow(list(xi) + [yi])
    print("csv write successfull!!")

def generate_imbalanced(n_samples: int, imbalance: float):
    print("running make_classification:")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        weights=[1 - imbalance, imbalance],
        flip_y=0.01,
        class_sep=1.0,
        random_state=42
    )

    print("make clasification successfull")

    print("now writing to csv:")

    header = [f"x_{i}" for i in range(X.shape[1])] + ["y"]
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for xi, yi in zip(X, y):
            writer.writerow(list(xi) + [yi])
    print("csv write successfull!!")

def write_csv(filename: str, X: np.ndarray, y: np.ndarray):
  
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_1", "x_2", "y"])
        for xi, yi in zip(X, y):
            writer.writerow([*xi, yi])


parser = argparse.ArgumentParser(description="Generate synthetic classification dataset(csv).")
parser.add_argument("--type",type=str, default="linear", help="Type of dataset: linear, noisy, high-dim-sparse")
parser.add_argument("--n_samples", type=int, default=200, help="Number of samples")
parser.add_argument("--center0", nargs=2, type=float, default=[-5, -5],help="Center of class 0")
parser.add_argument("--center1", nargs=2, type=float, default=[5, 5],help="Center of class 1")
parser.add_argument("--cov_scale", type=float, default=1.0,help="Cluster variance scale")
parser.add_argument("--n_features", type=int, default=100,help="Total number of features")
parser.add_argument("--n_informative", type=int, default=5, help="Number of truly predictive features")
parser.add_argument("--flip_y", type=float, default=0.01, help="Label noise fraction")
parser.add_argument("--imbalance", type=float, default=0.2,help="Fraction of samples in class 1 (e.g. 0.2 => 20%)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--output", type=str, default="data_classification.csv", help="Output CSV path")

args = parser.parse_args()
centers = [tuple(args.center0), tuple(args.center1)]
print("running generator")

if args.type == "noisy":
    print("running noisy generator")
    generate_noisy_classification_data(args.n_samples, args.seed, args.output)

elif args.type == "linear":
    print("running linear generator")
    X, y = generate_linear_separable(args.n_samples, centers, args.cov_scale, args.seed)
    write_csv(args.output, X, y)
    print(f"Linear dataset generated with {args.n_samples} samples to {args.output}")

elif args.type == "high-dim-sparse":
    print("running high-dim-sparse generator")
    generate_hig_dim_sparse( args.n_samples, args.n_features,args.n_informative,args.flip_y)
    print(f"High-dimensional sparse binary classification dataset generated with {args.n_samples} samples to {args.output}")

elif args.type == "imbalance":
    print("running imbalance generator")
    generate_imbalanced(args.n_samples, args.imbalance)
    print(f"Imbalanced binary classification binary classification dataset generated with {args.n_samples} samples to {args.output}")


