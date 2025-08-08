# importing libs:
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models

# Custom Dataset for Drone Video Data:
# dataset preparation:
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

# evaluation of metrics:
def evaluate(model, test_loader):
    model.eval()
    mse = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            mse += nn.functional.mse_loss(outputs, labels, reduction='sum').item()
    
    mse /= len(test_loader.dataset)
    print(f'Test MSE: {mse:.4f}')
    
    # Control-specific metrics
    throttle_mae = nn.L1Loss()(outputs[:,0], labels[:,0])
    print(f'Throttle MAE: {throttle_mae:.4f}')

# training the model:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DroneNavigationCNN(sequence_length=5).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

def train(model, dataloader, epochs=50):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}')
        
        # Save checkpoint
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

train(model, train_loader, epochs=50)

# Convert to TensorRT for NVIDIA Jetson
from torch2trt import torch2trt

model.eval()
example_input = torch.rand(1, 15, 224, 224).cuda()  # 5 frames * 3 channels
model_trt = torch2trt(model, [example_input], fp16_mode=True)

# Save optimized model
torch.save(model_trt.state_dict(), 'drone_navigation_trt.pth')


