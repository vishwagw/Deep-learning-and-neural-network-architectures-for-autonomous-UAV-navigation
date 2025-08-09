import cv2
import numpy as np

cap = cv2.VideoCapture('flight1.mp4')
frame_data = []
label_data = []

for ts, row in zip(video_timestamps, aligned_data):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize and normalize
    frame = cv2.resize(frame, (224, 224))
    frame_data.append(frame)
    label_data.append([row['throttle'], row['yaw'], 
                      row['pitch'], row['roll']])

np.savez('dataset.npz', frames=frame_data, labels=label_data)