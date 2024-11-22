import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.05, upper_quantile: float = 0.95):
        """
        A Winsorizer transformer to clip data based on specified quantiles.

        Parameters:
        - lower_quantile: The lower quantile threshold (default=0.05)
        - upper_quantile: The upper quantile threshold (default=0.95)
        """
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.lower_quantile_ = {}
        self.upper_quantile_ = {}

    def fit(self, X, y=None):
        """
        Calculate quantile bounds for each numeric column.

        Parameters:
        - X: Input data (Pandas DataFrame or array-like)
        - y: Ignored (compatibility with scikit-learn)

        Returns:
        - self: Fitted transformer.
        """
        X = pd.DataFrame(X)  # Ensure input is a DataFrame
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        X_Num = X.select_dtypes(include=numerics)  # Select numeric columns

        # Compute quantile bounds for each column
        for col in X_Num.columns:
            self.lower_quantile_[col] = X_Num[col].quantile(self.lower_quantile)
            self.upper_quantile_[col] = X_Num[col].quantile(self.upper_quantile)

        return self

    def transform(self, X):
        """
        Apply Winsorization to numeric columns based on stored quantile bounds.

        Parameters:
        - X: Input data (Pandas DataFrame or array-like)

        Returns:
        - Transformed data as a NumPy array.
        """
        X = pd.DataFrame(X)  # Ensure input is a DataFrame
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        X_Num = X.select_dtypes(include=numerics)  # Select numeric columns

        # Apply clipping for each numeric column
        for col in X_Num.columns:
            X[col] = X[col].clip(lower=self.lower_quantile_[col], upper=self.upper_quantile_[col])

        return X.values  # Return transformed data as a NumPy array
