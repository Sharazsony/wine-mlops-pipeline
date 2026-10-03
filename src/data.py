"""
src/data.py
-----------
Data loading and validation for Wine Cultivar Classification.
Loads sklearn Wine dataset and performs stratified 80/20 train-test split.
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20
EXPECTED_FEATURES = 13
EXPECTED_CLASSES = 3


def load_and_split():
    """
    Load the Wine dataset and return a stratified 80/20 train-test split.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
    feature_names : list of str
    """
    wine = load_wine()

    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")

    validate_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, wine.feature_names


def validate_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Run data-quality assertions on the raw dataset.

    Raises
    ------
    AssertionError if validation fails.
    """
    # No null values in features
    null_count = X.isnull().sum().sum()
    assert null_count == 0, (
        f"Found {null_count} null value(s) in feature matrix."
    )

    # No null values in labels
    assert y.isnull().sum() == 0, "Found null value(s) in target labels."

    # Correct feature count
    assert X.shape[1] == EXPECTED_FEATURES, (
        f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}."
    )

    # Correct number of target classes
    assert y.nunique() == EXPECTED_CLASSES, (
        f"Expected {EXPECTED_CLASSES} classes, got {y.nunique()}."
    )

    print(
        f"[data] Validation passed: {X.shape[0]} samples, "
        f"{X.shape[1]} features, {y.nunique()} classes."
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_split()
    print(f"[data] Train size: {X_train.shape}, Test size: {X_test.shape}")
