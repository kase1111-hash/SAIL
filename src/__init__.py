"""
SAIL - Situational Awareness Interaction Layer

A privacy-first, locally-hosted AI companion that provides contextual
guidance for everyday situations where knowledge gaps can lead to harm.
"""

__version__ = "0.1.0"
__author__ = "SAIL Contributors"

from .config import Config, get_config, load_config

__all__ = ["Config", "__version__", "get_config", "load_config"]
