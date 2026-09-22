# Gaze-Detection-Mouse
Lightweight TensorFlow Lite (.tflite) models derived from MediaPipe for real-time face detection and 478-point 3D face mesh processing

# MediaPipe TFLite Models Repository

This repository contains lightweight TensorFlow Lite (.tflite) model files derived from Google MediaPipe for face detection and face landmark extraction, along with Python inference scripts.

## Models Included

1. Face Detector (face_detector.tflite): Short-range front-facing camera face detection model.
2. Face Landmarks Detector (face_landmarks_detector.tflite): 478 3D face mesh landmark detector.

## Usage

### Install dependencies
pip install -r requirements.txt

### Run Face Detection
python setup_model.py
python gaze_mouse.py

## License
This project utilizes models from Google MediaPipe and is distributed under the Apache License 2.0.
