"""Resample non-16kHz WAV files to 16kHz in-place."""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import librosa
import soundfile as sf

TARGET_SR = 16000
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "Audio Dataset")

fixed = 0
skipped = 0

for root, dirs, files in os.walk(DATASET_ROOT):
    for fname in sorted(files):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(root, fname)
        rel = os.path.relpath(fpath, DATASET_ROOT)

        # Load with original sample rate
        y, sr = librosa.load(fpath, sr=None)

        if sr != TARGET_SR:
            # Resample to 16kHz
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            # Overwrite the original file
            sf.write(fpath, y_resampled, TARGET_SR, subtype='PCM_16')
            print(f"  FIXED: {rel}  ({sr} -> {TARGET_SR} Hz)")
            fixed += 1
        else:
            skipped += 1

print(f"\nDone! Resampled {fixed} files. {skipped} files were already at {TARGET_SR} Hz.")
