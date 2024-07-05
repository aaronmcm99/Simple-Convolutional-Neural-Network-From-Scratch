import numpy as np
import tensorflow as tf
import os

# Define the file path where you want to save the dataset
save_dir = "../mnist_data"
os.makedirs(save_dir, exist_ok=True)

# Load the dataset using TensorFlow/Keras
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

# Save the dataset to disk
np.save(os.path.join(save_dir, "train_images.npy"), train_images)
np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)
np.save(os.path.join(save_dir, "test_images.npy"), test_images)
np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)

print("MNIST dataset downloaded and saved to disk.")
