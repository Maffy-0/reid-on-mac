"""
Person detection and tracking using YOLOv8 and BoT-SORT.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from . import config
from .logger import logger


class PersonTracker:
    """Person detection and tracking using YOLOv8 and BoT-SORT."""
    
    def __init__(self):
        self.model_path = config.MODELS_DIR / config.YOLO_MODEL
        self.device = config.DEVICE
        self.conf_threshold = config.YOLO_CONF_THRESHOLD
        self.iou_threshold = config.YOLO_IOU_THRESHOLD
        self.tracker_config = config.TRACKER
        
        # Track management
        self.active_tracks: Dict[int, dict] = {}
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        
        # Initialize YOLO model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load YOLOv8 model with tracking capabilities."""
        try:
            # Download model if not exists
            model = YOLO(config.YOLO_MODEL)
            
            # Move to specified device
            model.to(self.device)
            
            logger.log_info(f"Loaded YOLO model on device: {self.device}")
            return model
            
        except Exception as e:
            logger.log_error("Failed to load YOLO model", e)
            raise
    
    def detect_and_track(self, frame: np.ndarray) -> List[dict]:
        """
        Detect and track persons in the frame.
        
        Returns:
            List of detection dictionaries with keys:
            - track_id: Track ID
            - bbox: [x1, y1, x2, y2]
            - confidence: Detection confidence
            - crop: Person crop image
        """
        try:
            # Run tracking with error handling for missing dependencies
            try:
                results = self.model.track(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    tracker=self.tracker_config,
                    classes=[0],  # Person class only
                    verbose=False
                )
            except ImportError as e:
                if "lap" in str(e):
                    logger.log_warning("BoT-SORT tracking unavailable (missing 'lap' package), using detection only")
                    # Fallback to detection only
                    results = self.model(
                        frame,
                        conf=self.conf_threshold,
                        classes=[0],
                        verbose=False
                    )
                else:
                    raise e
            
            detections = []
            
            # Check if tracking results have IDs (tracking mode) or not (detection only mode)
            if results[0].boxes is not None:
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    # Tracking mode with IDs
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        # Convert xywh to xyxy
                        x_center, y_center, width, height = box
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        # Ensure coordinates are within frame bounds
                        h, w = frame.shape[:2]
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        # Extract person crop
                        crop = frame[y1:y2, x1:x2]
                        
                        if crop.size > 0:
                            detection = {
                                'track_id': track_id,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'crop': crop
                            }
                            detections.append(detection)
                            
                            # Update track history
                            center_x = int(x_center)
                            center_y = int(y_center)
                            
                            if track_id not in self.track_history:
                                self.track_history[track_id] = []
                            self.track_history[track_id].append((center_x, center_y))
                            
                            # Limit history length
                            if len(self.track_history[track_id]) > 30:
                                self.track_history[track_id] = self.track_history[track_id][-30:]
                                
                            # Update active tracks
                            self.active_tracks[track_id] = {
                                'last_seen': len(self.track_history[track_id]),
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf)
                            }
                
                else:
                    # Detection only mode (no tracking IDs)
                    logger.log_warning("Running in detection-only mode (tracking not available)")
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Ensure coordinates are within frame bounds
                        h, w = frame.shape[:2]
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        # Extract person crop
                        crop = frame[y1:y2, x1:x2]
                        
                        if crop.size > 0:
                            # Use frame index as fake track ID for detection-only mode
                            fake_track_id = 1000 + i  # Start from 1000 to distinguish from real track IDs
                            
                            detection = {
                                'track_id': fake_track_id,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'crop': crop
                            }
                            detections.append(detection)
            
            # Clean up old tracks
            self._cleanup_old_tracks()
            
            return detections
            
        except Exception as e:
            logger.log_error("Detection and tracking failed", e)
            return []
    
    def _cleanup_old_tracks(self, max_age: int = 30):
        """Remove tracks that haven't been seen for a while."""
        tracks_to_remove = []
        
        for track_id, track_info in self.active_tracks.items():
            if track_info['last_seen'] > max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.active_tracks.pop(track_id, None)
            self.track_history.pop(track_id, None)
    
    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Get movement history for a track."""
        return self.track_history.get(track_id, [])
    
    def draw_tracking_info(self, frame: np.ndarray, detections: List[dict], 
                          person_ids: Dict[int, str]) -> np.ndarray:
        """Draw tracking information on frame."""
        frame_copy = frame.copy()
        
        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Get person ID
            person_id = person_ids.get(track_id, "Unknown")
            
            # Draw bounding box
            color = config.BBOX_COLOR
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw track history
            if track_id in self.track_history:
                points = self.track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame_copy, points[i-1], points[i], color, 2)
            
            # Draw label
            label = f"ID:{track_id} {person_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       config.TEXT_SCALE, config.TEXT_THICKNESS)[0]
            
            # Background for text
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, 
                       config.TEXT_COLOR, config.TEXT_THICKNESS)
        
        return frame_copy
