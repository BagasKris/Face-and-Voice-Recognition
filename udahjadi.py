import os
import cv2
import numpy as np
import librosa
import sounddevice as sd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import pickle
import time

def load_model(model_path="voice_recognition_model.pkl"):
    with open(model_path, "rb") as file:
        data = pickle.load(file)
    return data["model"], data["label_encoder"], data["scaler"]

def extract_features_from_audio(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def predict_speaker_from_audio(audio, sr, model, label_encoder, scaler):
    feature = extract_features_from_audio(audio, sr)
    feature_scaled = scaler.transform([feature])
    prediction = model.predict(feature_scaled)
    speaker = label_encoder.inverse_transform(prediction)
    return speaker[0]

def record_audio(duration=3, samplerate=22050):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()  
    return audio, samplerate

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Dataset/training.xml')

print("Loading voice recognition model...")
voice_model, label_encoder, scaler = load_model()

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

if not video.isOpened():
    print("Error: Kamera tidak dapat diakses!")
    exit()

speaker_name = ""
speaker_display_time = 0
SPEAKER_DISPLAY_DURATION = 5  

while True:
    check, frame = video.read()
    if not check:
        print("Error: Frame tidak terbaca!")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    detected_faces = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = gray[y:y+h, x:x+w]
        id, conf = recognizer.predict(face_img)
        
        if conf < 99:
            if id == 1:
                name = 'Bagas'
            elif id == 2:
                name = 'Bryan'
            elif id == 3:
                name = 'Kenneth'
            elif id == 4:
                name = 'Oswal'
            elif id == 5:
                name = 'Andi'
            elif id == 14:
                name = 'Gunawan'
            elif id == 6:
                name = 'Keyzia'
            else:
                name = 'Unknown'
        else:
            name = 'Unknown'

        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        detected_faces.append(name)

    if time.time() - speaker_display_time < SPEAKER_DISPLAY_DURATION:
        cv2.putText(frame, f"Speaker: {speaker_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Speaker:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face and Voice Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("Starting voice recognition...")
        try:
            audio, sr = record_audio()
            speaker_name = predict_speaker_from_audio(audio, sr, voice_model, label_encoder, scaler)
            print(f"Identified Speaker: {speaker_name}")
            speaker_display_time = time.time()
        except Exception as e:
            print(f"Error in voice recognition: {e}")
            speaker_name = "Error"
            speaker_display_time = time.time()

video.release()
cv2.destroyAllWindows()
