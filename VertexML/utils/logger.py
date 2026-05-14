"""Logging utilities for VertexML."""

import logging

logger = logging.getLogger("vertexml")
logging.basicConfig(level=logging.INFO)


def log(message):
    """Log a message."""
    logger.info(message)
