"""Clustering machine learning pipeline."""

import logging
from core.preprocessing.preprocessor import Preprocessor
from core.models.clustering import train_clusterer

logger = logging.getLogger('VertexML')


class ClusteringPipeline:
    """Pipeline for clustering workflows."""

    def run(self):
        """Run the clustering pipeline."""
        logger.info("Running VertexML clustering pipeline")
