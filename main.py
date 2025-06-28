#!/usr/bin/env python3
"""
YOLOv8 + BoT-SORT + FastReID Person Re-identification System

This system performs real-time person detection, tracking, and re-identification
using webcam input. It plays personalized audio when known persons are detected.

Features:
- Person detection using YOLOv8
- Multi-object tracking with BoT-SORT
- Person re-identification using feature matching
- Personalized audio playback
- Event logging to CSV

Author: Project MEW
Date: 2025-06-28
"""
import sys
import time
import threading
from typing import Dict, Set

import cv2
import numpy as np

from src import config
from src.logger import logger
from src.audio_manager import AudioManager
from src.person_reid import PersonReID
from src.person_tracker import PersonTracker


class PersonReIDSystem:
    """Main system class for person re-identification."""
    
    def __init__(self):
        self.running = False
        self.cap = None
        
        # Initialize components
        self.tracker = PersonTracker()
        self.reid = PersonReID()
        self.audio_manager = AudioManager()
        
        # State management
        self.current_person_ids: Dict[int, str] = {}  # track_id -> person_id
        self.last_audio_time: Dict[str, float] = {}  # person_id -> timestamp
        self.audio_cooldown = 5.0  # seconds
        
        # Entry/exit detection
        self.previous_tracks: Set[int] = set()
        
        logger.log_info("Person Re-ID System initialized")
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture."""
        try:
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            
            if not self.cap.isOpened():
                logger.log_error("Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
            
            logger.log_info(f"Camera initialized: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
            return True
            
        except Exception as e:
            logger.log_error("Camera initialization failed", e)
            return False
    
    def process_frame(self, frame: np.ndarray):
        """Process a single frame for detection, tracking, and re-identification."""
        # Detect and track persons
        detections = self.tracker.detect_and_track(frame)
        
        if not detections:
            return frame
        
        # Current tracks in this frame
        current_tracks = set(det['track_id'] for det in detections)
        
        # Detect new entries
        new_tracks = current_tracks - self.previous_tracks
        for track_id in new_tracks:
            logger.log_person_event(track_id, "Unknown", "entry")
        
        # Detect exits
        exited_tracks = self.previous_tracks - current_tracks
        for track_id in exited_tracks:
            person_id = self.current_person_ids.get(track_id, "Unknown")
            logger.log_person_event(track_id, person_id, "exit")
            
            # Stop audio for exited person
            if person_id != "Unknown":
                self.audio_manager.stop_person_audio(person_id)
            
            # Clean up
            self.current_person_ids.pop(track_id, None)
        
        # Update previous tracks
        self.previous_tracks = current_tracks.copy()
        
        # Process each detection for re-identification
        for detection in detections:
            track_id = detection['track_id']
            crop = detection['crop']
            confidence = detection['confidence']
            
            # Perform re-identification
            person_id, reid_confidence = self.reid.identify_person(crop)
            
            # Update person ID for this track
            previous_id = self.current_person_ids.get(track_id, "Unknown")
            self.current_person_ids[track_id] = person_id
            
            # Log re-identification event
            if person_id != "Unknown":
                logger.log_person_event(track_id, person_id, "identified", reid_confidence)
                
                # Play audio if person is newly identified or re-identified after a break
                current_time = time.time()
                last_played = self.last_audio_time.get(person_id, 0)
                
                if current_time - last_played > self.audio_cooldown:
                    self.audio_manager.play_person_audio(person_id)
                    self.last_audio_time[person_id] = current_time
        
        # Draw tracking and re-identification info
        annotated_frame = self.tracker.draw_tracking_info(
            frame, detections, self.current_person_ids
        )
        
        return annotated_frame
    
    def run_realtime(self):
        """Run real-time person re-identification system."""
        if not self.initialize_camera():
            return
        
        self.running = True
        logger.log_info("Starting real-time person re-identification")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.log_error("Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Person Re-ID System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.log_info("Quit signal received")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = config.DATA_DIR / f"frame_{timestamp}.jpg"
                    cv2.imwrite(str(filename), processed_frame)
                    logger.log_info(f"Frame saved: {filename}")
                elif key == ord('r'):
                    # Reset system state
                    self.reset_system()
                    logger.log_info("System state reset")
        
        except KeyboardInterrupt:
            logger.log_info("System interrupted by user")
        
        except Exception as e:
            logger.log_error("Unexpected error in main loop", e)
        
        finally:
            self.cleanup()
    
    def reset_system(self):
        """Reset system state."""
        self.current_person_ids.clear()
        self.last_audio_time.clear()
        self.previous_tracks.clear()
        self.audio_manager.stop_all_audio()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.audio_manager.stop_all_audio()
        
        logger.log_info("System cleanup completed")
    
    def add_person_template(self, person_id: str, image_path: str):
        """Add a new person template from image file."""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                self.reid.add_person_template(person_id, image)
                logger.log_info(f"Added template for {person_id} from {image_path}")
            else:
                logger.log_error(f"Failed to load image: {image_path}")
        except Exception as e:
            logger.log_error(f"Failed to add template for {person_id}", e)
    
    def list_registered_persons(self):
        """List all registered persons."""
        persons = self.reid.get_registered_persons()
        logger.log_info(f"Registered persons: {', '.join(persons)}")
        return persons


def print_usage():
    """Print usage information."""
    print("Person Re-identification System")
    print("===============================")
    print("Controls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  r - Reset system state")
    print("")
    print("Usage:")
    print("  uv run python main.py                                    # Run real-time system")
    print("  uv run python main.py --add-template PERSON_ID IMAGE     # Add person template")
    print("  uv run python main.py --list-templates                   # List all templates")
    print("  uv run python main.py --remove-person PERSON_ID          # Remove person")
    print("  uv run python main.py --template-info PERSON_ID          # Show template info")
    print("")


def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) == 2:
        if sys.argv[1] == "--list-templates":
            try:
                system = PersonReIDSystem()
                template_info = system.reid.list_person_templates()
                print("\nRegistered persons and template counts:")
                print("=" * 40)
                if template_info:
                    for person_id, count in template_info.items():
                        print(f"{person_id}: {count} templates")
                else:
                    print("No persons registered yet.")
                print()
            except Exception as e:
                print(f"Error listing templates: {e}")
                logger.log_error("Failed to list templates", e)
            return
        else:
            print_usage()
            return
    
    elif len(sys.argv) == 3:
        if sys.argv[1] == "--remove-person":
            person_id = sys.argv[2]
            try:
                system = PersonReIDSystem()
                if system.reid.remove_person_all_templates(person_id):
                    print(f"Removed all templates for {person_id}")
                else:
                    print(f"Person {person_id} not found")
            except Exception as e:
                print(f"Error removing person: {e}")
                logger.log_error(f"Failed to remove person {person_id}", e)
            return
            
        elif sys.argv[1] == "--template-info":
            person_id = sys.argv[2]
            try:
                system = PersonReIDSystem()
                count = system.reid.get_person_template_count(person_id)
                if count > 0:
                    print(f"{person_id}: {count} templates")
                else:
                    print(f"Person {person_id} not found")
            except Exception as e:
                print(f"Error getting template info: {e}")
                logger.log_error(f"Failed to get template info for {person_id}", e)
            return
        else:
            print_usage()
            return
    
    elif len(sys.argv) == 4:
        if sys.argv[1] == "--add-template":
            person_id = sys.argv[2]
            image_path = sys.argv[3]
            
            system = PersonReIDSystem()
            system.add_person_template(person_id, image_path)
            return
        else:
            print_usage()
            return
    
    elif len(sys.argv) > 1:
        print_usage()
        return
    
    # Check system requirements
    logger.log_info("Checking system requirements...")
    
    # Check PyTorch MPS availability
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.log_info("MPS (Metal Performance Shaders) is available")
        else:
            logger.log_warning("MPS is not available, using CPU")
    except ImportError:
        logger.log_error("PyTorch is not installed")
        return
    
    # Check camera availability
    test_cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not test_cap.isOpened():
        logger.log_error("Camera is not available")
        return
    test_cap.release()
    
    # Print usage information
    print_usage()
    
    # Start system
    system = PersonReIDSystem()
    
    # List registered persons
    system.list_registered_persons()
    
    # Run real-time system
    try:
        system.run_realtime()
    except Exception as e:
        logger.log_error("System failed", e)
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
