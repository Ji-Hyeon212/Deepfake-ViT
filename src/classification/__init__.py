"""
Classification module
"""

from .classifier import (
    MLPClassifier,
    AttentionClassifier,
    EnsembleClassifier
)

__all__ = [
    'MLPClassifier',
    'AttentionClassifier',
    'EnsembleClassifier'
]