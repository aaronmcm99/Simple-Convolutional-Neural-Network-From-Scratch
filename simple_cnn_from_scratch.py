import os
import numpy as np
import tensorflow as tf

from simple_cnn import SimpleCNN


# Load the dataset from disk
save_dir = "mnist_data"
x_train = np.load(os.path.join(save_dir, "train_images.npy"))
y_train = np.load(os.path.join(save_dir, "train_labels.npy"))
x_test = np.load(os.path.join(save_dir, "test_images.npy"))
y_test = np.load(os.path.join(save_dir, "test_labels.npy"))

# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension to the data
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Instantiate the CNN
cnn = SimpleCNN()

# Forward pass for a batch of images
batch_size = 32
flat_outputs = []

for i in range(0, len(x_train), batch_size):
    batch = x_train[i : i + batch_size]
    for image in batch:
        flat_output = cnn.forward(image)
        flat_outputs.append(flat_output)

flat_outputs = np.array(flat_outputs)

# Fully connected layer using TensorFlow
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(400,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
print("Starting training...")
for epoch in range(5):
    for i in range(0, len(flat_outputs), batch_size):
        batch_x = flat_outputs[i : i + batch_size]
        batch_y = y_train[i : i + batch_size]
        with tf.GradientTape() as tape:
            preds = model(batch_x)
            loss = tf.keras.losses.categorical_crossentropy(batch_y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        cnn.backward(grads, learning_rate=0.001)
        if i % 100 == 0:
            print(
                f"Epoch {epoch+1}, Step {i // batch_size}, Loss: {np.mean(loss.numpy())}"
            )

# Evaluate the model
flat_test_outputs = []
for image in x_test:
    flat_output = cnn.forward(image)
    flat_test_outputs.append(flat_output)

flat_test_outputs = np.array(flat_test_outputs)
test_loss, test_accuracy = model.evaluate(flat_test_outputs, y_test, verbose=2)

print(f"Test accuracy: {test_accuracy}")
