# Emotion-Detection-from-Soldier-Speech
Objective:

Detects the emotional state of a soldier from their speech audio
Classifies speech into 3 target emotions: Happy, Sad, and Angry
Built to assist in battlefield/military mental health monitoring scenarios


Dataset:

Uses the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song)
Audio from 24 actors (01–24), .wav format
8 emotion labels in the dataset: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
Only 3 are used for this project: Happy, Sad, Angry
Emotion label is encoded in the filename (3rd segment after splitting by -)


Feature Extraction:

Uses Librosa to load each audio file (3 seconds, 0.5s offset)
Extracts MFCC (Mel-Frequency Cepstral Coefficients) — 40 coefficients per file
MFCCs are averaged across time frames to produce a fixed-size 40-dimensional feature vector per audio sample


Preprocessing:

Features standardized using StandardScaler (zero mean, unit variance)
Labels encoded using LabelEncoder and converted to one-hot vectors using to_categorical
Data split: 80% training / 20% testing with random_state=42


Model Architecture:

Simple Feedforward Neural Network (Dense layers) built with TensorFlow/Keras
Layer 1: Dense(256, ReLU) + Dropout(0.3)
Layer 2: Dense(128, ReLU) + Dropout(0.3)
Output: Dense(3, Softmax) — one node per emotion class
Optimizer: Adam | Loss: Categorical Crossentropy | Metric: Accuracy


Training:

50 epochs, batch size of 16
Validation done on the test split during training


Evaluation:

Reports Test Accuracy on unseen data
Generates a Classification Report (Precision, Recall, F1-score per class)
Visualizes a Confusion Matrix using Seaborn heatmap


Output:

Trained model saved as emotion_detection_model.keras (also .h5 format present)
Model can be reloaded later for inference on new audio files


Tech Stack:

Python, TensorFlow/Keras, Librosa, Scikit-learn, NumPy, Seaborn, Matplotlib
