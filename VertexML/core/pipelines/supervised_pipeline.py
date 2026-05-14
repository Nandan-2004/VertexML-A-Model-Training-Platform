"""Supervised machine learning pipeline."""

import logging
from core.preprocessing.preprocessor import Preprocessor
from core.models.classification import train_classifier

logger = logging.getLogger('VertexML')


class SupervisedPipeline:
    """Pipeline for supervised learning."""

    def __init__(self):
        self.preprocessor = Preprocessor()

    def run(self):
        """Run the supervised pipeline."""
        logger.info("Running VertexML supervised pipeline")
