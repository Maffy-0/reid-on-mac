"""
Logging utilities for the person re-identification system.
"""
import csv
import logging
from datetime import datetime

from . import config


class PersonLogger:
    """Logger for person tracking and re-identification events."""
    
    def __init__(self):
        self.log_file = config.LOG_FILE
        self.setup_logging()
        self.setup_csv_log()
        
    def setup_logging(self):
        """Setup standard Python logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOGS_DIR / 'system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_csv_log(self):
        """Setup CSV log file with headers if not exists."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(config.LOG_HEADERS)
                
    def log_person_event(self, track_id: int, person_id: str, 
                        event_type: str, confidence: float = 0.0):
        """Log person tracking/re-identification event."""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, track_id, person_id, event_type, confidence])
            
        # Log to standard logger
        self.logger.info(f"Person Event: Track={track_id}, ID={person_id}, "
                        f"Event={event_type}, Confidence={confidence:.3f}")
        
    def log_info(self, message: str):
        """Log general information."""
        self.logger.info(message)
        
    def log_error(self, message: str, exception: Exception = None):
        """Log error message."""
        if exception:
            self.logger.error(f"{message}: {str(exception)}")
        else:
            self.logger.error(message)
            
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)


# Global logger instance
logger = PersonLogger()
