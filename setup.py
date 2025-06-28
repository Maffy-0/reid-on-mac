#!/usr/bin/env python3
"""
Setup script for creating necessary directories and sample files.
"""
import os
from pathlib import Path

from src import config
from src.logger import logger


def create_directories():
    """Create necessary directories."""
    directories = [
        config.MODELS_DIR,
        config.DATA_DIR,
        config.AUDIO_DIR,
        config.LOGS_DIR,
        config.TEMPLATES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def create_sample_audio_files():
    """Create sample audio files."""
    try:
        from pydub import AudioSegment
        pydub_available = True
    except ImportError:
        pydub_available = False
        
    sample_persons = ["person_1", "person_2", "person_3"]
    
    for person_id in sample_persons:
        audio_file = config.AUDIO_DIR / f"{person_id}.mp3"
        if not audio_file.exists():
            if pydub_available:
                try:
                    # Create a simple tone as sample audio
                    duration = 3000  # 3 seconds
                    silence = AudioSegment.silent(duration=duration)
                    
                    # Export as MP3
                    silence.export(str(audio_file), format="mp3")
                    print(f"Created sample audio: {audio_file}")
                except Exception as e:
                    # Create empty file if export fails
                    audio_file.touch()
                    print(f"Created empty audio file due to error: {audio_file}")
            else:
                # Create empty file if pydub not available
                audio_file.touch()
                print(f"Created empty audio file (pydub not available): {audio_file}")


def main():
    """Main setup function."""
    print("Setting up Person Re-identification System...")
    
    # Create directories
    create_directories()
    
    # Create sample audio files
    create_sample_audio_files()
    
    print("\nSetup completed!")
    print("\nNext steps:")
    print("1. Add your own audio files to the 'audio' directory")
    print("2. Add person templates using: python main.py --add-template PERSON_ID IMAGE_PATH")
    print("3. Run the system: python main.py")


if __name__ == "__main__":
    main()
