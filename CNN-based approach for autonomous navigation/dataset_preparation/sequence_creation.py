def create_sequences(frames, labels, seq_length=5):
    X, y = [], []
    for i in range(len(frames) - seq_length):
        X.append(frames[i:i+seq_length])
        y.append(labels[i+seq_length-1])  # Predict next action
    return np.array(X), np.array(y)