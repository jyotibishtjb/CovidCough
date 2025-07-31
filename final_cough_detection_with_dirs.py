
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# === Step 1: Feature Extraction ===
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        if y is None or len(y) < sr * 0.3:
            raise ValueError("Audio too short or empty")

        features = {
            'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'rmse': np.mean(librosa.feature.rms(y=y)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        }

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f'mfcc{i}'] = np.mean(mfccs[i - 1])

        return features
    except Exception as e:
        print(f"Skipping file {os.path.basename(file_path)}: {e}")
        return None

# === Step 2: Load Dataset from Original Directory ===
folder_path = "/Users/goodwill/.cache/kagglehub/datasets/himanshu007121/coughclassifier-trial/versions/22"
audio_dirs = ['MelSpectograms', 'trial_covid']

data = []

for subfolder in audio_dirs:
    subfolder_path = os.path.join(folder_path, subfolder)
    for file in os.listdir(subfolder_path):
        if file.endswith('.wav') or file.endswith('.mp3'):
            file_path = os.path.join(subfolder_path, file)
            features = extract_features(file_path)
            if features:
                features['filename'] = file
                features['source_folder'] = subfolder
                data.append(features)

df_features = pd.DataFrame(data)

# === Step 3: Merge with Labels ===
label_path = os.path.join(folder_path, 'cough_dataset.csv')
df_labels = pd.read_csv(label_path)

df = pd.merge(df_features, df_labels, left_on='filename', right_on='filename')
df = df.drop(columns=['filename', 'source_folder'])

# === Step 4: Preprocessing ===
X = df.drop(['label'], axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# === Step 5: Model Training ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 6: Evaluation ===
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# === Step 7: Predict New Sample ===
def predict_cough(file_path):
    features = extract_features(file_path)
    if features:
        df_sample = pd.DataFrame([features])
        df_sample_scaled = scaler.transform(df_sample)
        prediction = model.predict(df_sample_scaled)
        result = le.inverse_transform(prediction)[0]
        print(f"Prediction for '{os.path.basename(file_path)}':", result)
    else:
        print("Could not process audio.")

# Example: predict_cough('/Users/goodwill/Desktop/sample.wav')
