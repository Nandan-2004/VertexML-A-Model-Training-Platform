from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from typing import Sequence


def build_numeric_transformer() -> Pipeline:
    """Build a numeric transformation pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def build_categorical_transformer(use_dense: bool = True) -> Pipeline:
    """Build a categorical transformation pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=not use_dense))
    ])


def build_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str], use_dense: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer from numeric and categorical feature lists."""
    return ColumnTransformer([
        ('num', build_numeric_transformer(), list(numeric_features)),
        ('cat', build_categorical_transformer(use_dense), list(categorical_features))
    ], remainder='drop')


def build_feature_selector(threshold: float = 0.01) -> VarianceThreshold:
    """Build a variance threshold selector."""
    return VarianceThreshold(threshold=threshold)
