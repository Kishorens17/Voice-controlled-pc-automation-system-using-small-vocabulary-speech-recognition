"""Extract mel spectrograms from audio dataset and prepare train/val/test splits."""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# ======================== Config ========================
SR = 16000
DURATION = 2.5          # seconds (covers all samples: max was 2.31s)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
TARGET_SAMPLES = int(SR * DURATION)
FIXED_TIME_STEPS = int(np.ceil(TARGET_SAMPLES / HOP_LENGTH)) + 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "Audio Dataset")
OUTPUT_FILE = os.path.join(BASE_DIR, "processed_data.npz")

CLASSES = sorted([
    "Background", "Close_Window", "Increase_Volume", "Mute_Volume", "Nova",
    "Open_Chrome", "Open_Task_Manager", "Open_Visual_Studio_Code",
    "Reduce_Volume", "Shut_Down_System"
])


def extract_mel_spectrogram(file_path):
    """Load audio file and return a normalized mel spectrogram."""
    y, _ = librosa.load(file_path, sr=SR)

    # Pad or truncate to fixed duration
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]

    # Mel spectrogram -> dB scale
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max > mel_min:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_db = np.zeros_like(mel_db)

    # Ensure fixed time dimension
    if mel_db.shape[1] < FIXED_TIME_STEPS:
        mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_TIME_STEPS - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :FIXED_TIME_STEPS]

    return mel_db


def main():
    X, y = [], []
    class_counts = {c: 0 for c in CLASSES}

    print("Scanning audio files...")
    for root, _, files in os.walk(DATASET_ROOT):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(root, fname)
            class_name = os.path.basename(root)
            if class_name not in CLASSES:
                continue
            try:
                mel = extract_mel_spectrogram(fpath)
                X.append(mel)
                y.append(CLASSES.index(class_name))
                class_counts[class_name] += 1
            except Exception as e:
                print(f"  Error: {fpath}: {e}")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]   # (N, n_mels, time, 1)
    y = np.array(y, dtype=np.int32)

    print(f"\n--- Dataset Summary ---")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape[1:]}  (n_mels, time_steps, 1)")
    print(f"\nPer-class counts:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # Stratified split: 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        classes=CLASSES
    )

    print(f"\n--- Split ---")
    print(f"Train : {len(X_train)}  ({len(X_train)//len(CLASSES)} per class)")
    print(f"Val   : {len(X_val)}")
    print(f"Test  : {len(X_test)}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
