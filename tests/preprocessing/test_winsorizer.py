import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):
    # Create a sample dataset with a known distribution
    X = np.random.normal(0, 1, 1000)

    # Run the Winsorizer with the given quantiles
    winsorizer = Winsorizer(lower_quantile, upper_quantile)
    winsorizer.fit(X)
    X_transformed = winsorizer.transform(X)

    # Calculate the lower and upper bounds for the test
    lower_bound = np.percentile(X, lower_quantile * 100)
    upper_bound = np.percentile(X, upper_quantile * 100)

    # Assert that all transformed values are within the quantile bounds
    assert np.all(X_transformed >= lower_bound), f"Some values are below the lower quantile bound of {lower_bound}"
    assert np.all(X_transformed <= upper_bound), f"Some values are above the upper quantile bound of {upper_bound}"

    # Optional: Check that the transformed array has the same length as the original
    assert len(X_transformed) == len(X), "Transformed array length does not match the original array length"