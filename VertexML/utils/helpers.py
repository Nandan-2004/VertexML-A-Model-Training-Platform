"""General helper utilities for VertexML."""


def load_config(path):
    """Load a configuration file."""
    return {}


def save_json(path, data):
    """Save data as JSON."""
    import json

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
