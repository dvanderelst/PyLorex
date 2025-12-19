from pathlib import Path
from piper import PiperVoice
import pygame
import time
import wave


class SoundPlayer:

    def __init__(self):
        """
        Initialize the SoundPlayer with folders for MP3s and TTS voices,
        relative to the module's directory.
        """
        pygame.mixer.init()
        module_dir = Path(__file__).parent  # Get the directory where this module is located
        self.sound_folder = module_dir / 'sounds'
        self.voice_folder = module_dir / 'voices'
        self.temp_folder = module_dir / 'temp'
        # Create the temp folder if it doesn't exist
        self.temp_folder.mkdir(exist_ok=True)
        self.sounds = {}
        self.voice = None
        self._load_sounds()
        self._load_voice()

    def _load_sounds(self):
        """Load all MP3 files from the sound folder into memory."""
        if not self.sound_folder.exists():
            raise FileNotFoundError(f"Sound folder '{self.sound_folder}' not found.")
        for mp3_file in self.sound_folder.glob("*.mp3"):
            sound_name = mp3_file.stem
            self.sounds[sound_name] = pygame.mixer.Sound(str(mp3_file))

    def _load_voice(self):
        """Load the Piper TTS voice."""
        voice_path = self.voice_folder / "en_GB-jenny_dioco-medium.onnx"
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file '{voice_path}' not found.")
        self.voice = PiperVoice.load(str(voice_path))

    def play(self, sound_name, volume=1.0, blocking=True):
        """
        Play the sound with the given name (without the .mp3 extension).
        volume: float between 0.0 (silent) and 1.0 (full volume).
        blocking: if True, wait for the sound to finish before returning.
        """
        if sound_name in self.sounds:
            sound = self.sounds[sound_name]
            sound.set_volume(volume)
            sound.play()
            if blocking:
                time.sleep(sound.get_length())
        else:
            # Treat as TTS if not an MP3
            self.speak(sound_name, volume, blocking)

    def speak(self, text, volume=1.0, blocking=True):
        """
        Speak the given text using Piper TTS.
        """
        if not self.voice:
            raise RuntimeError("TTS voice not loaded.")
        temp_wav = self.temp_folder / "temp.wav"
        with wave.open(str(temp_wav), "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        sound = pygame.mixer.Sound(str(temp_wav))
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


# import pygame
# import time
# from pathlib import Path
# import wave
# from os import path
# from piper import PiperVoice, set
#
# class Speaker:
#     def __init__(self):
#         self.temp_folder = 'Temp'
#         self.voice_folder = 'Voices'
#         self.voice = PiperVoice.load(path.join(self.voice_folder, 'en_GB-jenny_dioco-medium.onnx'))
#
#     def synthesize(self, text, wave_file=None):
#         if wave_file is None: wave_file = path.join(self.temp_folder, 'Temp.wav')
#         with wave.open(wave_file, "wb") as wav_file:
#             self.voice.synthesize_wav(text, wav_file)
#
#     def stream(self, text):
#         for chunk in self.voice.synthesize(text):
#             set_audio_format(chunk.sample_rate, chunk.sample_width, chunk.sample_channels)
#             write_raw_data(chunk.audio_int16_bytes)
#
#
#
#
# class SoundPlayer:
#     def __init__(self, sound_folder="sounds"):
#         """
#         Initialize the SoundPlayer with a folder containing MP3 files.
#         Each file should be named like 'shutter.mp3', so you can call s.play('shutter').
#         """
#         pygame.mixer.init()
#         self.sound_folder = Path(sound_folder)
#         self.sounds = {}
#         self._load_sounds()
#
#     def _load_sounds(self):
#         """Load all MP3 files from the sound folder into memory."""
#         if not self.sound_folder.exists():
#             raise FileNotFoundError(f"Sound folder '{self.sound_folder}' not found.")
#
#         for mp3_file in self.sound_folder.glob("*.mp3"):
#             sound_name = mp3_file.stem  # e.g., 'shutter' for 'shutter.mp3'
#             self.sounds[sound_name] = pygame.mixer.Sound(str(mp3_file))
#
#     def play(self, sound_name, volume=1.0, blocking=True):
#         """
#         Play the sound with the given name (without the .mp3 extension).
#         volume: float between 0.0 (silent) and 1.0 (full volume).
#         blocking: if True, wait for the sound to finish before returning.
#         """
#         if sound_name not in self.sounds:
#             raise ValueError(f"Sound '{sound_name}' not found.")
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
