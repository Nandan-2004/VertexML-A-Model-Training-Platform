import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from typing import List, Tuple


def detect_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return a series with the count of missing values per column."""
    return df.isna().sum()


def drop_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop duplicate rows and return the cleaned DataFrame with the duplicate count."""
    duplicates = int(df.duplicated().sum())
    return df.drop_duplicates(), duplicates


def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop constant columns and return the cleaned DataFrame plus removed column names."""
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    return df.drop(columns=constant_cols, errors='ignore'), constant_cols


def drop_high_cardinality_columns(df: pd.DataFrame, threshold: int = 50) -> Tuple[pd.DataFrame, List[str]]:
    """Drop categorical columns with cardinality above the threshold."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    high_card_cols = [col for col in categorical_cols if df[col].nunique(dropna=False) > threshold]
    return df.drop(columns=high_card_cols, errors='ignore'), high_card_cols


def impute_missing_values(df: pd.DataFrame, numeric_strategy: str = 'median', categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
    """Impute missing values for numeric and categorical columns."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy=numeric_strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df
