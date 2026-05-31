"""Check audio dataset consistency: sample rate, channels, duration, bit depth."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import wave
import contextlib
from collections import defaultdict

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "Audio Dataset")

stats = defaultdict(list)
errors = []
file_count = 0

for root, dirs, files in os.walk(DATASET_ROOT):
    for fname in sorted(files):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(root, fname)
        rel = os.path.relpath(fpath, DATASET_ROOT)
        file_count += 1
        try:
            with contextlib.closing(wave.open(fpath, "r")) as w:
                sr = w.getframerate()
                ch = w.getnchannels()
                sw = w.getsampwidth()
                frames = w.getnframes()
                dur = frames / sr
                stats["sample_rate"].append((sr, rel))
                stats["channels"].append((ch, rel))
                stats["bit_depth"].append((sw * 8, rel))
                stats["duration"].append((dur, rel))
        except Exception as e:
            errors.append((rel, str(e)))

print(f"Total WAV files scanned: {file_count}\n")

if errors:
    print(f"⚠ {len(errors)} file(s) could not be read:")
    for rel, err in errors:
        print(f"   {rel}: {err}")
    print()

# --- Sample Rate ---
sr_set = set(v for v, _ in stats["sample_rate"])
print(f"Sample rates found: {sorted(sr_set)} Hz")
if len(sr_set) > 1:
    print("  ⚠ MISMATCH! Breakdown:")
    for sr in sorted(sr_set):
        files = [r for v, r in stats["sample_rate"] if v == sr]
        print(f"    {sr} Hz: {len(files)} files")
else:
    print("  ✓ All files have the same sample rate.")

# --- Channels ---
ch_set = set(v for v, _ in stats["channels"])
print(f"\nChannels found: {sorted(ch_set)}")
if len(ch_set) > 1:
    print("  ⚠ MISMATCH! Breakdown:")
    for ch in sorted(ch_set):
        files = [r for v, r in stats["channels"] if v == ch]
        print(f"    {ch} ch: {len(files)} files")
else:
    print("  ✓ All files have the same channel count.")

# --- Bit Depth ---
bd_set = set(v for v, _ in stats["bit_depth"])
print(f"\nBit depths found: {sorted(bd_set)}")
if len(bd_set) > 1:
    print("  ⚠ MISMATCH! Breakdown:")
    for bd in sorted(bd_set):
        files = [r for v, r in stats["bit_depth"] if v == bd]
        print(f"    {bd}-bit: {len(files)} files")
else:
    print("  ✓ All files have the same bit depth.")

# --- Duration ---
durs = [v for v, _ in stats["duration"]]
if durs:
    min_d, max_d, avg_d = min(durs), max(durs), sum(durs) / len(durs)
    print(f"\nDuration range: {min_d:.2f}s – {max_d:.2f}s  (avg {avg_d:.2f}s)")
    short = [(d, r) for d, r in stats["duration"] if d < 0.3]
    long_ = [(d, r) for d, r in stats["duration"] if d > 5.0]
    if short:
        print(f"  ⚠ {len(short)} file(s) shorter than 0.3s:")
        for d, r in short[:5]:
            print(f"    {r}: {d:.2f}s")
    if long_:
        print(f"  ⚠ {len(long_)} file(s) longer than 5.0s:")
        for d, r in long_[:5]:
            print(f"    {r}: {d:.2f}s")
    if not short and not long_:
        print("  ✓ All durations look reasonable.")

# --- Per-class file counts ---
print("\n--- Files per class (Original) ---")
orig_dir = None
for root, dirs, files in os.walk(DATASET_ROOT):
    if os.path.basename(root) == "Original" or "Original" in root:
        parent = os.path.basename(os.path.dirname(root)) if os.path.basename(root) != "Original" else None
        # Only list leaf class folders
        wavs = [f for f in files if f.lower().endswith(".wav")]
        if wavs:
            cls_name = os.path.basename(root)
            print(f"  {cls_name}: {len(wavs)} samples")

print("\n--- Files per class (Augmented) ---")
for root, dirs, files in os.walk(DATASET_ROOT):
    if "Augmented" in root:
        wavs = [f for f in files if f.lower().endswith(".wav")]
        if wavs:
            cls_name = os.path.basename(root)
            print(f"  {cls_name}: {len(wavs)} samples")

print("\n✅ Consistency check complete.")
