import os

DATA_PATH = r"E:\Project\Emotion Detection from Soldier Speech\data"
print(f"🔍 Checking files in: {DATA_PATH}")

wav_count = 0
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith('.wav'):
            print(f"🎧 Found: {os.path.join(root, file)}")
            wav_count += 1

print(f"✅ Total .wav files found: {wav_count}")
