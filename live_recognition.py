import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import librosa
import pyaudio
import wave
import subprocess
import threading
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

# ======================== Config ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "speech_crnn.keras")
TEMP_WAV = os.path.join(BASE_DIR, "last_recording.wav")

SR = 16000
DURATION = 2.5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
TARGET_SAMPLES = int(SR * DURATION)
FIXED_TIME_STEPS = int(np.ceil(TARGET_SAMPLES / HOP_LENGTH)) + 1

CONFIDENCE_THRESHOLD = 0.40
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

CLASSES = sorted([
    "Background", "Close_Window", "Increase_Volume", "Mute_Volume", "Nova",
    "Open_Chrome", "Open_Task_Manager", "Open_Visual_Studio_Code",
    "Reduce_Volume", "Shut_Down_System"
])


# ======================== Attention Layer (for model loading) ========================
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


# ======================== Feature Extraction ========================
def extract_mel_spectrogram(audio_data):
    """Convert raw audio (numpy float32) to a normalized mel spectrogram."""
    y = audio_data.copy()

    # Pad or truncate
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max > mel_min:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_db = np.zeros_like(mel_db)

    if mel_db.shape[1] < FIXED_TIME_STEPS:
        mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_TIME_STEPS - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :FIXED_TIME_STEPS]

    return mel_db


# ======================== System Commands ========================
def execute_command(command_name):
    """Execute the system action corresponding to the recognized command."""
    print(f"  >> Executing: {command_name}")

    try:
        if command_name == "Close_Window":
            import pyautogui
            pyautogui.hotkey('alt', 'F4')

        elif command_name == "Increase_Volume":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            current = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(min(1.0, current + 0.1), None)
            print(f"     Volume: {min(1.0, current + 0.1):.0%}")

        elif command_name == "Reduce_Volume":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            current = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(max(0.0, current - 0.1), None)
            print(f"     Volume: {max(0.0, current - 0.1):.0%}")

        elif command_name == "Mute_Volume":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            is_muted = volume.GetMute()
            volume.SetMute(not is_muted, None)
            print(f"     Mute: {'ON' if not is_muted else 'OFF'}")

        elif command_name == "Open_Chrome":
            subprocess.Popen(["start", "chrome"], shell=True)

        elif command_name == "Open_Task_Manager":
            subprocess.Popen(["taskmgr.exe"])

        elif command_name == "Open_Visual_Studio_Code":
            subprocess.Popen(["code"], shell=True)


        elif command_name == "Shut_Down_System":
            # To enable: uncomment the line below (This will shutdown your PC immediately)
            # subprocess.Popen(["shutdown", "/s", "/t", "1"])
            print("     !!! SHUTDOWN REQUESTED — (commented out for safety) !!!")


        elif command_name == "Nova":
            print("     Wake word detected!")

        elif command_name == "Background":
            pass  # No action for background noise

    except Exception as e:
        print(f"     Error executing {command_name}: {e}")


# ======================== Recording ========================
def record_audio(duration=DURATION):
    """Record audio from the microphone and return as float32 numpy array."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=SR,
                     input=True, frames_per_buffer=CHUNK)

    frames = []
    num_chunks = int(SR * duration / CHUNK)

    for _ in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Convert to float32
    audio = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return audio


# ======================== Main Loop ========================
def main():
    print("=" * 60)
    print("  LIVE SPEECH COMMAND RECOGNITION")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {MODEL_FILE}")
    model = keras.models.load_model(
        MODEL_FILE, custom_objects={'Attention': Attention}
    )
    print("Model loaded successfully!\n")

    print(f"Classes: {CLASSES}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"Recording duration: {DURATION}s")
    print(f"\nPress Ctrl+C to stop.\n")
    print("-" * 60)

    try:
        while True:
            input("Press ENTER to record (Ctrl+C to quit)...")
            print("  Recording...", end=" ", flush=True)

            audio = record_audio()
            print("Done!")

            # Extract features
            mel = extract_mel_spectrogram(audio)
            mel_input = mel[np.newaxis, ..., np.newaxis]   # (1, n_mels, time, 1)

            # Predict
            probs = model.predict(mel_input, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            pred_class = CLASSES[pred_idx]

            # Display results
            print(f"  Prediction: {pred_class}  ({confidence:.1%})")

            # Show top 3
            top3 = np.argsort(probs)[::-1][:3]
            for rank, idx in enumerate(top3, 1):
                bar = "#" * int(probs[idx] * 30)
                print(f"    {rank}. {CLASSES[idx]:30s} {probs[idx]:6.1%}  {bar}")

            # Execute if confident enough
            if confidence >= CONFIDENCE_THRESHOLD and pred_class != "Background":
                execute_command(pred_class)
            elif confidence < CONFIDENCE_THRESHOLD:
                print(f"  (Below threshold {CONFIDENCE_THRESHOLD:.0%} — no action)")

            print("-" * 60)

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
