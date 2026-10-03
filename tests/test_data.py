"""
tests/test_data.py
------------------
Unit tests for src/data.py — data loading and validation.
"""

import pytest
from src.data import load_and_split, validate_data, EXPECTED_FEATURES, EXPECTED_CLASSES


class TestLoadAndSplit:
    """Tests for load_and_split()."""

    def setup_method(self):
        self.X_train, self.X_test, self.y_train, self.y_test, self.features = load_and_split()

    def test_train_test_shapes(self):
        """Total samples should equal 178 (Wine dataset size)."""
        total = len(self.X_train) + len(self.X_test)
        assert total == 178, f"Expected 178 total samples, got {total}"

    def test_split_ratio(self):
        """Test split should be approximately 20% of total."""
        ratio = len(self.X_test) / (len(self.X_train) + len(self.X_test))
        assert 0.18 <= ratio <= 0.22, f"Test ratio {ratio:.2f} outside [0.18, 0.22]"

    def test_feature_count(self):
        """Feature matrix must have exactly 13 columns."""
        assert self.X_train.shape[1] == EXPECTED_FEATURES
        assert self.X_test.shape[1] == EXPECTED_FEATURES

    def test_no_nulls_in_features(self):
        """No null values allowed in features."""
        assert self.X_train.isnull().sum().sum() == 0
        assert self.X_test.isnull().sum().sum() == 0

    def test_no_nulls_in_labels(self):
        """No null values allowed in labels."""
        assert self.y_train.isnull().sum() == 0
        assert self.y_test.isnull().sum() == 0

    def test_label_classes(self):
        """Labels must only contain {0, 1, 2}."""
        all_labels = set(self.y_train.unique()) | set(self.y_test.unique())
        assert all_labels == {0, 1, 2}, f"Unexpected classes: {all_labels}"

    def test_stratification(self):
        """Both splits should contain all 3 classes."""
        assert self.y_train.nunique() == EXPECTED_CLASSES
        assert self.y_test.nunique() == EXPECTED_CLASSES

    def test_reproducibility(self):
        """Two calls must return identical splits (fixed random_state=42)."""
        X_tr2, X_te2, y_tr2, y_te2, _ = load_and_split()
        assert (self.X_train.values == X_tr2.values).all()
        assert (self.y_test.values == y_te2.values).all()

    def test_feature_names_returned(self):
        """feature_names must be a non-empty sequence of 13 strings."""
        assert len(self.features) == EXPECTED_FEATURES
        assert all(isinstance(f, str) for f in self.features)


class TestValidateData:
    """Tests for validate_data() — should raise on bad input."""

    def setup_method(self):
        self.X_train, _, self.y_train, _, _ = load_and_split()

    def test_valid_data_passes(self):
        """Valid data should not raise."""
        validate_data(self.X_train, self.y_train)

    def test_null_in_features_raises(self):
        """Null in features should raise AssertionError."""
        X_bad = self.X_train.copy()
        X_bad.iloc[0, 0] = None
        with pytest.raises(AssertionError, match="null"):
            validate_data(X_bad, self.y_train)

    def test_wrong_feature_count_raises(self):
        """Wrong feature count should raise AssertionError."""
        X_bad = self.X_train.iloc[:, :10]  # only 10 features
        with pytest.raises(AssertionError, match="features"):
            validate_data(X_bad, self.y_train)
