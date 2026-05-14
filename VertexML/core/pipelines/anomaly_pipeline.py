"""Anomaly detection pipeline."""

import logging
from core.preprocessing.preprocessor import Preprocessor
from core.models.anomaly import train_anomaly_detector

logger = logging.getLogger('VertexML')


class AnomalyPipeline:
    """Pipeline for anomaly detection workflows."""

    def run(self):
        """Run the anomaly detection pipeline."""
        logger.info("Running VertexML anomaly detection pipeline")
