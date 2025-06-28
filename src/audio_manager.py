"""
Audio management for person-specific music playbook using pydub.
"""
import threading
import time
import subprocess
from pathlib import Path
from typing import Dict

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from . import config
from .logger import logger


class AudioManager:
    """Manages audio playback for person-specific music using system audio."""
    
    def __init__(self):
        self.audio_dir = config.AUDIO_DIR
        self.volume = config.AUDIO_VOLUME
        self.current_playing: Dict[str, threading.Thread] = {}
        self.stop_flags: Dict[str, threading.Event] = {}
        
        self._setup_audio_files()
        logger.log_info(f"Audio Manager initialized (pydub available: {PYDUB_AVAILABLE})")
        
    def _setup_audio_files(self):
        """Setup sample audio files if they don't exist."""
        sample_persons = ["person_1", "person_2", "person_3"]
        
        for person in sample_persons:
            audio_file = self.audio_dir / f"{person}.mp3"
            if not audio_file.exists():
                if PYDUB_AVAILABLE:
                    # Create a simple silence audio file as placeholder
                    try:
                        silence = AudioSegment.silent(duration=3000)  # 3 seconds
                        silence.export(str(audio_file), format="mp3")
                        logger.log_info(f"Created placeholder audio file: {audio_file}")
                    except Exception as e:
                        # Create empty file if export fails
                        audio_file.touch()
                        logger.log_warning(f"Created empty audio file due to error: {audio_file}")
                else:
                    # Create empty file if pydub not available
                    audio_file.touch()
                    logger.log_info(f"Created empty audio file: {audio_file}")
                
    def play_person_audio(self, person_id: str):
        """Play audio associated with a person using system audio player."""
        if person_id == "Unknown":
            return
            
        audio_file = self.audio_dir / f"{person_id}.mp3"
        if not audio_file.exists():
            logger.log_warning(f"Audio file not found for {person_id}")
            return
            
        # Stop current audio for this person if playing
        self.stop_person_audio(person_id)
        
        # Play using system's default audio player
        self._play_with_system_player(person_id, audio_file)
            
    def _play_with_system_player(self, person_id: str, audio_file: Path):
        """Play audio using macOS system audio player (afplay)."""
        def play_thread():
            try:
                # Create stop flag for this person
                stop_flag = threading.Event()
                self.stop_flags[person_id] = stop_flag
                
                # Use macOS built-in afplay command
                process = subprocess.Popen(
                    ["afplay", str(audio_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Monitor for stop signal
                while process.poll() is None:
                    if stop_flag.is_set():
                        process.terminate()
                        break
                    time.sleep(0.1)
                
                # Wait for process to complete
                process.wait()
                
                logger.log_info(f"Finished playing audio for {person_id}")
                
            except FileNotFoundError:
                # Fallback: try using system's open command
                try:
                    process = subprocess.Popen(
                        ["open", str(audio_file)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    process.wait()
                except Exception as e:
                    logger.log_error(f"Failed to play audio for {person_id}", e)
            except Exception as e:
                logger.log_error(f"Failed to play audio for {person_id}", e)
            finally:
                # Cleanup
                self.current_playing.pop(person_id, None)
                self.stop_flags.pop(person_id, None)
                
        thread = threading.Thread(target=play_thread, daemon=True)
        self.current_playing[person_id] = thread
        thread.start()
        
        logger.log_info(f"Started playing audio for {person_id}")
        
    def stop_person_audio(self, person_id: str):
        """Stop audio for a specific person."""
        # Signal stop flag
        if person_id in self.stop_flags:
            self.stop_flags[person_id].set()
            
        # Wait for thread to finish (with timeout)
        if person_id in self.current_playing:
            thread = self.current_playing[person_id]
            thread.join(timeout=1.0)  # Wait up to 1 second
            
    def stop_all_audio(self):
        """Stop all currently playing audio."""
        for person_id in list(self.current_playing.keys()):
            self.stop_person_audio(person_id)
            
    def is_playing(self, person_id: str) -> bool:
        """Check if audio is currently playing for a person."""
        return person_id in self.current_playing and self.current_playing[person_id].is_alive()
