# Simple-Convolutional-Neural-Network-From-Scratch
 This repository contains an implementation of a Convolutional Neural Network (CNN) from scratch, designed to work with the MNIST dataset. After previously creating a fully connected neural network from scratch, I decided to extend my work by implementing the convolutional and pooling layers (both forward and backward passes) from scratch. For efficiency, I used TensorFlow for the fully connected layer. You can find my previous project here: https://github.com/aaronmcm99/NeuralNetworkFromScratch

## Code Breakdown

### Importing Required Libraries

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
