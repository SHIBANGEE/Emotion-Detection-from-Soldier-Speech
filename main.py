import os
import librosa
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURATION ===
DATA_PATH = r"E:\\Project\\Emotion Detection from Soldier Speech\\data\\audio_speech_actors_01-24"
TARGET_EMOTIONS = ["happy", "sad", "angry"]  # Try a few first

# === EMOTION MAPPING ===
EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):
    
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        audio = librosa.util.normalize(audio)

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)

        features = np.hstack([
            np.mean(mfccs.T, axis=0),
            np.std(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.std(chroma.T, axis=0),
            np.mean(zcr.T),
            np.std(zcr.T)
        ])
        return features
    except Exception as e:
        print(f"⚠️ Could not process {file_path}: {e}")
        return None

def load_data(data_path, target_emotions=None):
    print("🔄 Loading and processing data...")
    x, y = [], []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                parts = file.split("-")
                if len(parts) < 3:
                    continue
                emotion_code = parts[2]
                label = EMOTION_LABELS.get(emotion_code)
                if label is None:
                    continue

                if target_emotions and label not in target_emotions:
                    continue

                features = extract_features(file_path)
                if features is not None:
                    x.append(features)
                    y.append(label)

    print(f"✅ Samples Loaded: {len(x)}")
    print(f"🎯 Class Distribution: {Counter(y)}")
    return np.array(x), np.array(y)

# === LOAD & PREPROCESS ===
x, y = load_data(DATA_PATH, target_emotions=TARGET_EMOTIONS)
scaler = StandardScaler()
x = scaler.fit_transform(x)

encoder = LabelEncoder()
y_encoded = to_categorical(encoder.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# === MODEL ===
model = Sequential([
    Dense(256, activation='relu', input_shape=(x.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(TARGET_EMOTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAIN ===
print("🚀 Training model...")
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=2
)

# === EVALUATE ===
loss, acc = model.evaluate(x_test, y_test)
print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")
model.save("emotion_detection_model.keras")
print("💾 Model saved as emotion_detection_model.keras")

# === CONFUSION MATRIX ===
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
