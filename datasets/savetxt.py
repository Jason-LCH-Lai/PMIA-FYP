import numpy as np

# Load your npy file
your_array = np.load('cluster_labels_100.npy', allow_pickle=True)

# Print the array
print("Original NumPy array:")
print(your_array)

# Reshape to 2D if it's not already
if your_array.ndim > 2:
    your_array = your_array.reshape(your_array.shape[0], -1)

# Save the array to a text file
np.savetxt('cluster_labels_100.txt', your_array)

# Print a message indicating successful save
print("NumPy array saved to 'your_array.txt'")