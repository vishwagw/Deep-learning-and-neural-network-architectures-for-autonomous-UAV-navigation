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