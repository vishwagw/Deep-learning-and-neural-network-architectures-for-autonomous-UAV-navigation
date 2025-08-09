class DroneDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        frame = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            frame = self.transform(image=frame)['image']
            
        return torch.from_numpy(frame.transpose(2,0,1)), torch.tensor(label)

train_dataset = DroneDataset(X_train, y_train, transform=aug)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)