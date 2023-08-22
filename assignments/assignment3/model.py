import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.out_classes = n_output_classes
        image_width, image_height, in_channels = input_shape
        self.layers = []
        self.layers.append(ConvolutionalLayer(in_channels, conv1_channels, 3, 1))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(4, 4))
        self.layers.append(ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(4, 4))
        self.layers.append(Flattener())
        self.layers.append(FullyConnectedLayer(4 * conv2_channels, n_output_classes))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        predictions = self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            predictions = self.layers[i].forward(predictions)
        loss, grad = softmax_with_cross_entropy(predictions, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss

    def predict(self, X):
        predictions = self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            predictions = self.layers[i].forward(predictions)
        probs = softmax(predictions)
        return np.argmax(probs, axis=1)

    def params(self):
        return {'Conv1.W': self.layers[0].W, 'Conv1.B': self.layers[0].B,
                'Conv2.W': self.layers[3].W, 'Conv2.B': self.layers[3].B,
                'FC.W': self.layers[7].W, 'FC.B': self.layers[7].B}
