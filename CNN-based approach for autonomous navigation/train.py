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