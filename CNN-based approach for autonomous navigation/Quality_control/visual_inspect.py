import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.imshow(X_train[0])
plt.title(f'Throttle: {y_train[0][0]:.2f}, Yaw: {y_train[0][1]:.2f}')
plt.show()