from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import NotFittedError

from .data_cleaner import (
    detect_missing_values,
    drop_constant_columns,
    drop_duplicate_rows,
    drop_high_cardinality_columns,
)
from .encoder import fit_label_encoder, transform_labels
from .feature_engineering import (
    build_feature_selector,
    build_preprocessor,
)


@st.cache_data(show_spinner="Loading file...")
def load_large_file(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext in ['csv', 'txt']:
            return pd.read_csv(uploaded_file)
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(uploaded_file)
        elif ext == 'parquet':
            return pd.read_parquet(uploaded_file)
        elif ext == 'feather':
            return pd.read_feather(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
    return None


class EnhancedUniversalDataPreprocessor:
    def __init__(self, use_dense=True, enable_feature_selection=True):
        self.label_encoder = None
        self.preprocessor = None
        self.feature_selector = build_feature_selector()
        self.feature_names = None
        self.use_dense = use_dense
        self.enable_feature_selection = enable_feature_selection
        self._fitted = False
        self._dropped_constant_cols = []
        self._dropped_high_card_cols = []
        self._feature_columns = None
        self.task_type_ = None
        self._variance_threshold_used = False

    def detect_task_type(self, y=None):
        if y is None:
            return 'clustering'
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            return 'regression'
        return 'classification'

    def _validate_data(self, df: pd.DataFrame, fit: bool = False):
        df_clean = df.copy()

        if fit or not self._fitted:
            df_clean, self._dropped_constant_cols = drop_constant_columns(df_clean)
            if self._dropped_constant_cols:
                st.info(f"Removing constant columns: {self._dropped_constant_cols}")

            df_clean, self._dropped_high_card_cols = drop_high_cardinality_columns(df_clean)
            if self._dropped_high_card_cols:
                st.info(f"Removing high cardinality features: {self._dropped_high_card_cols}")

            df_clean, duplicates = drop_duplicate_rows(df_clean)
            if duplicates > 0:
                st.info(f"Removing {duplicates} duplicate rows")

        return df_clean

    def _build_preprocessor(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        return build_preprocessor(numeric_features, categorical_features, self.use_dense)

    def process(self, df: pd.DataFrame, target_col: Optional[str] = None, task_type_override: Optional[str] = None):
        df_clean = self._validate_data(df, fit=True)

        if target_col is None:
            X = df_clean.copy()
            y = None
            task_type = 'clustering'
        else:
            X = df_clean.drop(columns=[target_col])
            y = df_clean[target_col]

            valid_indices = y.notna()
            X = X[valid_indices]
            y = y[valid_indices]

            task_type = task_type_override if task_type_override in ['classification', 'regression'] else self.detect_task_type(y)

            if task_type == 'regression' and not pd.api.types.is_numeric_dtype(y):
                st.error("Regression selected but target column is not numeric. Choose Classification or use a numeric target.")
                raise ValueError("Non-numeric target cannot be used for regression.")

            if task_type == 'classification':
                self.label_encoder = fit_label_encoder(y)
                y = transform_labels(self.label_encoder, y)

        final_indices = X.index
        self._feature_columns = list(X.columns)
        self.task_type_ = task_type

        self.preprocessor = self._build_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)

        original_shape = X_processed.shape[1]
        self._variance_threshold_used = False
        if self.enable_feature_selection and original_shape > 10:
            try:
                self.feature_selector.fit(X_processed)
                X_processed = self.feature_selector.transform(X_processed)
                self._variance_threshold_used = True
                new_shape = X_processed.shape[1]
                if new_shape < original_shape:
                    st.info(f"Feature selection: Reduced from {original_shape} to {new_shape} features")
            except Exception as e:
                st.warning(f"Feature selection skipped: {str(e)}")

        try:
            cat_names = self.preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(X.select_dtypes(include=['object', 'category']).columns)
        except (AttributeError, KeyError):
            cat_names = []

        self.feature_names = list(X.select_dtypes(include=np.number).columns) + list(cat_names)

        try:
            if self._variance_threshold_used and self.feature_selector.get_support().size == len(self.feature_names):
                self.feature_names = np.array(self.feature_names)[self.feature_selector.get_support()].tolist()
        except Exception:
            pass

        self._fitted = True
        return X_processed, y, task_type, final_indices

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None, task_type_override: Optional[str] = None):
        if not self._fitted or self.preprocessor is None:
            raise NotFittedError("Preprocessor is not fitted. Call process(...) first.")

        df_clean = self._validate_data(df, fit=False)

        if target_col is None:
            X = df_clean.copy()
            y = None
            task_type = self.task_type_ or 'clustering'
        else:
            if target_col not in df_clean.columns:
                raise KeyError(f"Target column '{target_col}' not found in provided data.")

            X = df_clean.drop(columns=[target_col])
            y = df_clean[target_col]

            valid_indices = y.notna()
            X = X[valid_indices]
            y = y[valid_indices]

            task_type = task_type_override if task_type_override in ['classification', 'regression'] else (self.task_type_ or self.detect_task_type(y))

            if task_type == 'classification' and self.label_encoder is not None:
                try:
                    y = transform_labels(self.label_encoder, y)
                except ValueError:
                    known = set(self.label_encoder.classes_)
                    mask = y.isin(known)
                    dropped = int((~mask).sum())
                    if dropped > 0:
                        st.warning(
                            f"Dropped {dropped} rows from evaluation because they contain unseen target labels "
                            f"not present in the training split."
                        )
                    X = X[mask]
                    y = y[mask]
                    y = transform_labels(self.label_encoder, y)

            if task_type == 'regression' and not pd.api.types.is_numeric_dtype(y):
                st.error("Regression selected but target column is not numeric.")
                raise ValueError("Non-numeric target cannot be used for regression.")

        if self._feature_columns is not None:
            missing_cols = [c for c in self._feature_columns if c not in X.columns]
            if missing_cols:
                for col in missing_cols:
                    X[col] = np.nan
            X = X.reindex(columns=self._feature_columns)

        indices = X.index
        X_processed = self.preprocessor.transform(X)

        if self._variance_threshold_used:
            try:
                X_processed = self.feature_selector.transform(X_processed)
            except Exception:
                pass

        return X_processed, y, task_type, indices


@st.cache_data(show_spinner="Performing intelligent preprocessing...")
def get_processed_data(df: pd.DataFrame, target_col: Optional[str] = None, enable_feature_selection: bool = True):
    preprocessor = EnhancedUniversalDataPreprocessor(
        use_dense=True,
        enable_feature_selection=enable_feature_selection
    )
    X, y, task_type, indices = preprocessor.process(df.copy(), target_col)
    return {
        'X': X,
        'y': y,
        'task_type': task_type,
        'indices': indices,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names
    }
