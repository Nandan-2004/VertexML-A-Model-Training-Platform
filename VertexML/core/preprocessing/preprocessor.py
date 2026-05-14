"""Data preprocessing pipeline for VertexML."""

from .data_cleaner import clean_missing_values, remove_duplicates
from .encoder import encode_categorical


class Preprocessor:
    """Preprocessor for raw datasets."""

    def fit_transform(self, data):
        data = clean_missing_values(data)
        data = remove_duplicates(data)
        data = encode_categorical(data)
        return data
