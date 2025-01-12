import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Extract MFCC features from audio files
def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Step 2: Load dataset
def load_dataset(data_path):
    features = []
    labels = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)  # Folder name as label
                
                try:
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return np.array(features), np.array(labels)

# Step 3: Train model
def train_model(features, labels):
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

    # Train SVM classifier
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    return model, le, scaler

# Step 4: Save model to disk
def save_model(model, label_encoder, scaler, model_path="voice_recognition_model.pkl"):
    with open(model_path, "wb") as file:
        pickle.dump({"model": model, "label_encoder": label_encoder, "scaler": scaler}, file)

if __name__ == "__main__":
    # Set the path to your dataset (organized by folders with speaker names)
    dataset_path = "D:\Kenneth Contee\Kuliah\Semester 7\Pengolahan citra suara\Voice\dataset"  # Replace with the path to your dataset

    # Load dataset
    print("Loading dataset...")
    features, labels = load_dataset(dataset_path)

    # Train model
    print("Training model...")
    model, label_encoder, scaler = train_model(features, labels)

    # Save model
    save_model(model, label_encoder, scaler)
    print("Model saved successfully.")
