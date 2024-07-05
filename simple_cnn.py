import numpy as np


class SimpleCNN:
    def __init__(self):
        # Initialize filters for two convolutional layers
        self.conv1_filters = np.random.randn(6, 1, 5, 5) * 0.1
        self.conv2_filters = np.random.randn(16, 6, 5, 5) * 0.1

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    # Perform 2D convolution
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

    # Backpropagation for convolutional layer
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

    # Perform max pooling
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

    # Backpropagation for max pooling
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

    # Flatten the tensor
    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

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
