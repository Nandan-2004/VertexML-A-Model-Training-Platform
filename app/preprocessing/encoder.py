import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Optional


def fit_label_encoder(y: pd.Series) -> LabelEncoder:
    """Fit a label encoder on the target series."""
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder


def transform_labels(encoder: LabelEncoder, y: pd.Series) -> pd.Series:
    """Transform labels using a fitted LabelEncoder."""
    return pd.Series(encoder.transform(y), index=y.index, name=y.name)


def build_onehot_encoder(use_dense: bool = True) -> OneHotEncoder:
    """Build a OneHotEncoder for categorical features."""
    return OneHotEncoder(handle_unknown='ignore', sparse_output=not use_dense)
