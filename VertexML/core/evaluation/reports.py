"""Reporting utilities for VertexML."""


def generate_classification_report(y_true, y_pred):
    """Return classification report."""
    return {}


def export_report(report, path):
    """Export evaluation report to disk."""
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(str(report))
