"""Evaluation metrics and analysis tools."""

# Lazy imports to avoid eagerly loading heavy dependencies (nltk, pycocotools,
# HuggingFace evaluate, matplotlib, etc.) when the evaluation subpackage is imported.


def __getattr__(name: str):
    """Lazy import of evaluation classes."""
    _lazy_imports = {
        "CaptioningMetrics": ".metrics",
        "EvaluationRunner": ".metrics",
    }

    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CaptioningMetrics", "EvaluationRunner"]