"""
Vision processing package.

This package contains all image understanding components including
Gemini-based scene analysis, object cropping, embeddings, and matching.
"""

from .schemas import (
    BoundingBox,
    DetectedObject,
    SceneMeta,
    ImageAnalysis,
)

from .analyzer import GeminiAnalyzer

__all__ = [
    "BoundingBox",
    "DetectedObject",
    "SceneMeta",
    "ImageAnalysis",
    "GeminiAnalyzer",
]