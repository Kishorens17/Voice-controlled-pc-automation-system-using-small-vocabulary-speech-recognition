"""
Backend service for Nova: Handles microphone recording, 
feature extraction, and model inference.
"""
import os
import numpy as np
import librosa
import pyaudio
import wave
import time
import subprocess
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

# ======================== Configuration ========================
SR = 16000
DURATION = 2.5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
TARGET_SAMPLES = int(SR * DURATION)
FIXED_TIME_STEPS = int(np.ceil(TARGET_SAMPLES / HOP_LENGTH)) + 1

CLASSES = sorted([
    "Background", "Close_Window", "Increase_Volume", "Mute_Volume", "Nova",
    "Open_Chrome", "Open_Task_Manager", "Open_Visual_Studio_Code",
    "Reduce_Volume", "Shut_Down_System"
])

# ======================== Custom Layers ========================
class Attention(keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_w', shape=(input_shape[-1], 1),
            initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(
            name='att_b', shape=(input_shape[1], 1),
            initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

# ======================== Voice Service Class ========================
class VoiceService:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Loading Nova model from {model_path}...")
        self.model = keras.models.load_model(
            model_path, custom_objects={'Attention': Attention}
        )
        self.pa = pyaudio.PyAudio()
        self.is_listening = False

    def extract_mel(self, audio_data):
        y = audio_data.copy()
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
        else:
            y = y[:TARGET_SAMPLES]

        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize
        m_min, m_max = mel_db.min(), mel_db.max()
        if m_max > m_min:
            mel_db = (mel_db - m_min) / (m_max - m_min)
        else:
            mel_db = np.zeros_like(mel_db)

        # Pad time steps if necessary
        if mel_db.shape[1] < FIXED_TIME_STEPS:
            mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_TIME_STEPS - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :FIXED_TIME_STEPS]

        return mel_db[np.newaxis, ..., np.newaxis]

    def record_chunk(self):
        stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SR,
                             input=True, frames_per_buffer=1024)
        frames = []
        for _ in range(int(SR * DURATION / 1024)):
            frames.append(stream.read(1024, exception_on_overflow=False))
        
        stream.stop_stream()
        stream.close()
        return np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

    def predict(self, audio):
        mel = self.extract_mel(audio)
        probs = self.model.predict(mel, verbose=0)[0]
        idx = np.argmax(probs)
        return CLASSES[idx], probs[idx]

    def execute_action(self, command):
        """Execute system commands."""
        try:
            if command == "Close_Window":
                import pyautogui
                pyautogui.hotkey('alt', 'F4')
            elif command == "Increase_Volume":
                self._change_volume(0.1) # +10%
            elif command == "Reduce_Volume":
                self._change_volume(-0.1) # -10%
            elif command == "Mute_Volume":
                self._toggle_mute()
            elif command == "Open_Chrome":
                subprocess.Popen(["start", "chrome"], shell=True)
            elif command == "Open_Task_Manager":
                subprocess.Popen(["taskmgr.exe"])
            elif command == "Open_Visual_Studio_Code":
                subprocess.Popen(["code"], shell=True)

            elif command == "Shut_Down_System":
                # subprocess.Popen(["shutdown", "/s", "/t", "1"])
                pass

        except Exception as e:
            print(f"Action error: {e}")

    def _change_volume(self, delta):
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        curr = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, curr + delta)), None)

    def _toggle_mute(self):
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(not volume.GetMute(), None)

    def get_random_greeting(self):
        greetings = [
            "Nova here!, how can I help you?",
            "Yes? I'm listening.",
            "What do you need?",
            "At your service!",
            "I'm all ears!"
        ]
        return random.choice(greetings)
