import pygame
import time
from pathlib import Path

class SoundPlayer:
    def __init__(self, sound_folder="sounds"):
        """
        Initialize the SoundPlayer with a folder containing MP3 files.
        Each file should be named like 'shutter.mp3', so you can call s.play('shutter').
        """
        pygame.mixer.init()
        self.sound_folder = Path(sound_folder)
        self.sounds = {}
        self._load_sounds()

    def _load_sounds(self):
        """Load all MP3 files from the sound folder into memory."""
        if not self.sound_folder.exists():
            raise FileNotFoundError(f"Sound folder '{self.sound_folder}' not found.")

        for mp3_file in self.sound_folder.glob("*.mp3"):
            sound_name = mp3_file.stem  # e.g., 'shutter' for 'shutter.mp3'
            self.sounds[sound_name] = pygame.mixer.Sound(str(mp3_file))

    def play(self, sound_name, volume=1.0, blocking=True):
        """
        Play the sound with the given name (without the .mp3 extension).
        volume: float between 0.0 (silent) and 1.0 (full volume).
        blocking: if True, wait for the sound to finish before returning.
        """
        if sound_name not in self.sounds:
            raise ValueError(f"Sound '{sound_name}' not found.")
        sound = self.sounds[sound_name]
        sound.set_volume(volume)
        sound.play()
        if blocking:
            time.sleep(sound.get_length())

    def stop(self, sound_name=None):
        """
        Stop a specific sound or all sounds if no name is provided.
        """
        if sound_name:
            if sound_name in self.sounds:
                self.sounds[sound_name].stop()
        else:
            pygame.mixer.stop()

    def __del__(self):
        """Clean up pygame resources when the object is destroyed."""
        pygame.mixer.quit()
