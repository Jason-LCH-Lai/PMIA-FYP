import numpy as np

# Load the arrays
features = np.load('modified_purchase100.npy', allow_pickle=True)
labels = np.load('cluster_labels_100.npy', allow_pickle=True)

# Convert to Python lists
features_list = features.tolist()
labels_list = labels.tolist()

# Now you can use len()
num_samples_features = len(features_list['features'])
num_samples_labels = len(labels_list)

print(f"Number of samples in features: {num_samples_features}")
print(f"Number of samples in labels: {num_samples_labels}")
