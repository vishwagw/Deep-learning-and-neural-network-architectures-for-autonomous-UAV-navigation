import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class DroneDataset(Dataset):
    def __init__(self, video_paths, label_path, sequence_length=5, transform=None):
        """
        video_paths: List of paths to video files
        label_path: CSV with timestamps and control commands
        sequence_length: Number of consecutive frames to stack
        """
        self.transform = transform
        self.sequence_length = sequence_length
        self.labels = pd.read_csv(label_path)
        self.frames = []
        self.frame_labels = []
        
        # Extract frames and synchronize with labels
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Find nearest label in CSV
                label_row = self.labels.iloc[(self.labels['timestamp'] - timestamp).abs().argsort()[0]]
                self.frames.append(frame)
                self.frame_labels.append([label_row['throttle'], label_row['yaw'], 
                                         label_row['pitch'], label_row['roll']])
            cap.release()
    
    def __len__(self):
        return len(self.frames) - self.sequence_length
    
    def __getitem__(self, idx):
        frame_sequence = self.frames[idx:idx+self.sequence_length]
        labels = np.array(self.frame_labels[idx+self.sequence_length-1], dtype=np.float32)
        
        if self.transform:
            frame_sequence = [self.transform(frame) for frame in frame_sequence]
            
        # Stack frames along channel dimension
        return torch.cat(frame_sequence, dim=1), torch.tensor(labels)

# Example transformations
transform = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Usage
dataset = DroneDataset(video_paths=['flight1.mp4', 'flight2.mp4'], 
                      label_path='flight_labels.csv',
                      transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)