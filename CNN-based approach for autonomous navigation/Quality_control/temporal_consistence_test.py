for i in range(0, len(X_seq), 100):
    print(f"Sequence {i}: Throttle change = {y_seq[i+1][0] - y_seq[i][0]:.2f}")