from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    frames, labels, test_size=0.2, random_state=42)

# For sequence data
X_seq, y_seq = create_sequences(X_train, y_train)