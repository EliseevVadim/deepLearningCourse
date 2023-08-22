import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = []
        self.layers.append(FullyConnectedLayer(n_input=n_input, n_output=hidden_layer_size))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(n_input=hidden_layer_size, n_output=n_output))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        for param in self.params():
            self.params()[param].grad = np.zeros_like(self.params()[param].value)

        # forward run
        prediction = X.copy()
        for layer in self.layers:
            prediction = layer.forward(prediction)
        loss, grad = softmax_with_cross_entropy(prediction, y)
        # backward run
        l2 = 0
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_l2 = 0
            for params in layer.params():
                param = layer.params()[params]
                d_loss, d_grad = l2_regularization(param.value, self.reg)
                param.grad += d_grad
                l2 += d_loss
            grad += grad_l2
        loss += l2
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)
        return np.argmax(pred, axis=1)

    def params(self):
        result = {}
        for i in range(len(self.layers)):
            for j in self.layers[i].params():
                result[f"layer_{i}-param_{j}"] = self.layers[i].params()[j]
        return result
