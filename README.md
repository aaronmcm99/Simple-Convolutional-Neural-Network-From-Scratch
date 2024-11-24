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
### simple_cnn.py
#### SimpleCNN Class Breakdown
This class defines a simple Convolutional Neural Network (CNN) from scratch, including methods for forward and backward passes through the network.
#### Constructor (__init__ method)
- Purpose: Initializes the filters for two convolutional layers with random values.
- Explanation:
 - self.conv1_filters: Initializes filters for the first convolutional layer with a shape of (6, 1, 5, 5). These filters are initialized with random values sampled from a normal distribution (np.random.randn) and scaled by 0.1.
 - self.conv2_filters: Initializes filters for the second convolutional layer with a shape of (16, 6, 5, 5) in a similar manner.
```python
class SimpleCNN:
    def __init__(self):
        # Initialize filters for two convolutional layers
        self.conv1_filters = np.random.randn(6, 1, 5, 5) * 0.1
        self.conv2_filters = np.random.randn(16, 6, 5, 5) * 0.1
```

#### Activation Functions (relu, relu_derivative, softmax)
- Purpose: Implements activation functions used in the network.
- Explanation:
 - relu(self, x): Applies the ReLU (Rectified Linear Unit) activation function to input x, returning element-wise maximum of 0 and x.
 - relu_derivative(self, x): Computes the derivative of ReLU function with respect to x, returning 1 where x > 0 and 0 otherwise.
 - softmax(self, x): Computes the softmax activation function for input x, ensuring the output values sum up to 1 across each batch.
```python
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
```
#### Convolution Operations (conv2d, conv2d_backward)
- Purpose: Implements 2D convolution and its backward propagation.
- Explanation:
 - conv2d(self, x, filters): Performs 2D convolution on input x with given filters. It computes the dot product of each filter with the corresponding region of x, producing the output feature map.
 - conv2d_backward(self, d_out, x, filters): Computes gradients of x and filters with respect to the loss (d_out) during backpropagation. It distributes gradients from d_out back through the convolutional layer, updating d_x and d_filters.
```python
    def conv2d(self, x, filters):
        batch_size, in_channels, in_height, in_width = x.shape
        num_filters, _, filter_height, filter_width = filters.shape
        out_height = in_height - filter_height + 1
        out_width = in_width - filter_width + 1
        output = np.zeros((batch_size, num_filters, out_height, out_width))

        # Convolve each filter over the input batch
        for b in range(batch_size):
            for f in range(num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        region = x[b, :, i : i + filter_height, j : j + filter_width]
                        output[b, f, i, j] = np.sum(region * filters[f])
        return output

    def conv2d_backward(self, d_out, x, filters):
        batch_size, in_channels, in_height, in_width = x.shape
        num_filters, _, filter_height, filter_width = filters.shape
        d_x = np.zeros_like(x)
        d_filters = np.zeros_like(filters)

        # Iterate through each element of the gradient output
        for b in range(batch_size):
            for f in range(num_filters):
                for i in range(in_height - filter_height + 1):
                    for j in range(in_width - filter_width + 1):
                        region = x[b, :, i : i + filter_height, j : j + filter_width]
                        d_filters[f] += d_out[b, f, i, j] * region
                        d_x[b, :, i : i + filter_height, j : j + filter_width] += (
                            d_out[b, f, i, j] * filters[f]
                        )
        return d_x, d_filters
```

#### Pooling Operations (maxpool2d, maxpool2d_backward)
- Purpose: Implements max pooling and its backward propagation.
- Explanation:
 - maxpool2d(self, x, size=2, stride=2): Performs max pooling on input x with specified size and stride. It computes the maximum value within each pooling region, reducing the spatial dimensions of x.
 - maxpool2d_backward(self, d_out, x, size=2, stride=2): Computes gradients of x with respect to the loss (d_out) during backpropagation through max pooling. It distributes gradients to the corresponding locations of the original input x, updating d_x.
```python
    def maxpool2d(self, x, size=2, stride=2):
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - size) // stride + 1
        out_width = (in_width - size) // stride + 1
        output = np.zeros((batch_size, in_channels, out_height, out_width))

        # Apply max pooling
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = x[
                            b,
                            c,
                            i * stride : i * stride + size,
                            j * stride : j * stride + size,
                        ]
                        output[b, c, i, j] = np.max(region)
        return output

    def maxpool2d_backward(self, d_out, x, size=2, stride=2):
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - size) // stride + 1
        out_width = (in_width - size) // stride + 1
        d_x = np.zeros_like(x)

        # Distribute gradient to the corresponding max locations
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = x[
                            b,
                            c,
                            i * stride : i * stride + size,
                            j * stride : j * stride + size,
                        ]
                        max_val = np.max(region)
                        for m in range(size):
                            for n in range(size):
                                if region[m, n] == max_val:
                                    d_x[b, c, i * stride + m, j * stride + n] = d_out[
                                        b, c, i, j
                                    ]
        return d_x
```

#### Utility Methods (flatten)
- Purpose: Implements flattening of the tensor x.
- Explanation:
 - flatten(self, x): Reshapes the input x into a flattened shape, suitable for feeding into fully connected layers. It preserves the batch size (x.shape[0]) while flattening the remaining dimensions into a single dimension.
```python
    def flatten(self, x):
        return x.reshape(x.shape[0], -1)
```

#### Forward and Backward Passes (forward, backward)
- Purpose: Implements forward and backward passes for the eniter network.
- Explanation:
 - forward(self, x): Performs the forward propagation through the convolutional and pooling layers of the CNN. It first applies the first convolutional layer (conv1) to the input x, followed by ReLU activation and max pooling. Then, it applies the second convolutional layer (conv2) to the pooled output, again followed by ReLU activation and max pooling. Finally, it flattens the pooled output into a 1D vector for further processing or classification tasks.
 - backward(self, x): Computes gradients during backpropagation to update the convolutional filters (conv1_filters and conv2_filters) based on the gradient d_out of the loss function. It begins by reshaping d_out to match the shape of the second pooling layer (self.p2). It then propagates gradients backwards through each layer: applying the max pooling gradient, the ReLU derivative, and the convolutional layer gradient updates sequentially. The method adjusts the filters using the specified learning rate to optimize the network's performance over successive epochs.
 - ```python
   # Forward pass through the network
    def forward(self, x):
        self.x1 = self.conv2d(x, self.conv1_filters)
        self.a1 = self.relu(self.x1)
        self.p1 = self.maxpool2d(self.a1)

        self.x2 = self.conv2d(self.p1, self.conv2_filters)
        self.a2 = self.relu(self.x2)
        self.p2 = self.maxpool2d(self.a2)

        self.flat = self.flatten(self.p2)
        return self.flat

    # Backward pass through the network
    def backward(self, d_out, learning_rate=0.001):
        # Backprop through the fully connected layers
        d_p2 = d_out.reshape(self.p2.shape)

        # Backprop through the second maxpool layer
        d_a2 = self.maxpool2d_backward(d_p2, self.a2)

        # Backprop through the second ReLU layer
        d_x2 = d_a2 * self.relu_derivative(self.x2)

        # Backprop through the second conv layer
        d_p1, d_conv2_filters = self.conv2d_backward(d_x2, self.p1, self.conv2_filters)

        # Backprop through the first maxpool layer
        d_a1 = self.maxpool2d_backward(d_p1, self.a1)

        # Backprop through the first ReLU layer
        d_x1 = d_a1 * self.relu_derivative(self.x1)

        # Backprop through the first conv layer
        _, d_conv1_filters = self.conv2d_backward(d_x1, self.x, self.conv1_filters)

        # Update filters
        self.conv1_filters -= learning_rate * d_conv1_filters
        self.conv2_filters -= learning_rate * d_conv2_filters
   ```










