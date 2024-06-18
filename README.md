# Simple-Convolutional-Neural-Network-From-Scratch
 This repository contains an implementation of a Convolutional Neural Network (CNN) from scratch, designed to work with the MNIST dataset. After previously creating a fully connected neural network from scratch, I decided to extend my work by implementing the convolutional and pooling layers (both forward and backward passes) from scratch. For efficiency, I used TensorFlow for the fully connected layer. You can find my previous project here: https://github.com/aaronmcm99/NeuralNetworkFromScratch
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), with each image being a 28x28 pixel square. It is a foundational dataset in the machine learning community, often used for training and testing image processing systems. The goal is to correctly classify each image into one of the 10 digit classes.
The goal of this network is to correctly identify each greyscale numerical handwritten digit it receives as input.

## Code Breakdown

### simple_cnn_from_scratch.py
This file is used to load the training and test data, and construct, train and evaluate the CNN.

#### Importing Required Libraries
The code begins by importing necessary libraries:
```python
import os
import numpy as np
import tensorflow as tf
from simple_cnn import SimpleCNN
```
- os: Used for file path manipulations.
- numpy: Utilized for numerical operations.
- tensorflow: Employed for high-level neural network operations.
- SimpleCNN: Custom class defined in simple_cnn.py for the CNN implementation.
  
#### Loading the Dataset
The MNIST dataset is loaded from saved .npy files located in the directory named mnist_data.
```python
# Load the dataset from disk
save_dir = "mnist_data"
x_train = np.load(os.path.join(save_dir, "train_images.npy"))
y_train = np.load(os.path.join(save_dir, "train_labels.npy"))
x_test = np.load(os.path.join(save_dir, "test_images.npy"))
y_test = np.load(os.path.join(save_dir, "test_labels.npy"))
```

#### Normalizing the Data
The pixel values of the images are normalized to the range [0, 1] by dividing by 255.0.
```python
# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

#### Adding Channel Dimension
Since the images are grayscale, a channel dimension is added to match the expected input shape for the CNN.
```python
# Add channel dimension to the data
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)
```

#### Converting Labels to One-Hot Encoding
The labels are converted to one-hot encoding, which is the standard format for classification tasks.
```python
# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

#### Instantiating the CNN
An instance of the SimpleCNN class is created, which provides a custom implementation of the CNN.
```python
# Instantiate the CNN
cnn = SimpleCNN()
```

#### Forward Pass for Image Batches
This section performs the forward pass of the Convolutional Neural Network (CNN) on a batch of images (x_train). Each image is processed through the CNN (cnn.forward(image)) to obtain flattened outputs (flat_output). These outputs are collected into flat_outputs, which will be used for training the fully connected layer.
```python
# Forward pass for a batch of images
batch_size = 32
flat_outputs = []

for i in range(0, len(x_train), batch_size):
    batch = x_train[i : i + batch_size]
    for image in batch:
        flat_output = cnn.forward(image)
        flat_outputs.append(flat_output)

flat_outputs = np.array(flat_outputs)
```

#### Fully Connected Layer Using TensorFlow
Here, a fully connected neural network model is defined using TensorFlow's Keras API. The model consists of an input layer expecting 400-dimensional input (flattened CNN outputs), a hidden layer with 128 neurons and ReLU activation, and an output layer with 10 neurons (corresponding to 10 classes in MNIST) and softmax activation. The model is compiled with Adam optimizer, categorical cross-entropy loss function, and accuracy metric.
```python
# Fully connected layer using TensorFlow
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(400,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

#### Training the Model
This segment trains the fully connected layer using the flattened CNN outputs (flat_outputs) as input (batch_x). For each epoch, it iterates over batches (batch_size) of training data, computes predictions (preds), calculates the loss using categorical cross-entropy, computes gradients (grads), and updates the model parameters using the Adam optimizer. Additionally, it performs backpropagation through the CNN (cnn.backward(grads, learning_rate=0.001)).
```python
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
```

#### Evaluating the Model
Finally, this section evaluates the trained model on the test set (x_test) after converting each image using the CNN (cnn.forward(image)). It calculates the test loss and accuracy metrics using TensorFlow's evaluation function (model.evaluate), and prints the test accuracy.
```python
# Evaluate the model
flat_test_outputs = []
for image in x_test:
    flat_output = cnn.forward(image)
    flat_test_outputs.append(flat_output)

flat_test_outputs = np.array(flat_test_outputs)
test_loss, test_accuracy = model.evaluate(flat_test_outputs, y_test, verbose=2)

print(f"Test accuracy: {test_accuracy}")
```














