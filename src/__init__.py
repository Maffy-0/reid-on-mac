"""
Person Re-identification System - Core Package

This package contains the core modules for the YOLOv8 + BoT-SORT + FastReID
person re-identification system.
"""

__version__ = "0.1.0"
__author__ = "Project MEW"

# Import main components for easy access
from .config import *
from .logger import logger
from .person_tracker import PersonTracker
from .person_reid import PersonReID
from .audio_manager import AudioManager

__all__ = [
    'logger',
    'PersonTracker', 
    'PersonReID',
    'AudioManager'
]
