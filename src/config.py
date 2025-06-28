"""
Configuration settings for the person re-identification system.
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = PROJECT_ROOT / "audio"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Ensure directories exist
for dir_path in [MODELS_DIR, DATA_DIR, AUDIO_DIR, LOGS_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)

# YOLO settings
YOLO_MODEL = "yolov8n.pt"  # Lightweight model
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45

# Device settings
DEVICE = "mps" if os.uname().machine.startswith('arm') else "cpu"

# Tracking settings
TRACKER = "botsort.yaml"

# ReID settings
REID_SIMILARITY_THRESHOLD = 0.02  # Very low threshold for simple CNN features
REID_FEATURE_DIM = 2048
REID_MAX_TEMPLATES_PER_PERSON = 5  # Maximum templates per person
REID_SIMILARITY_METHOD = "max"  # "max", "avg", or "weighted"

# Audio settings
AUDIO_VOLUME = 0.8
AUDIO_FADE_DURATION = 1.0

# Video settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Display settings
BBOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

# Logging settings
LOG_FILE = LOGS_DIR / "person_log.csv"
LOG_HEADERS = ["timestamp", "track_id", "person_id", "event_type", "confidence"]
