import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    if predictions.ndim == 1:
        predictions -= np.max(predictions)
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds)
    else:
        predictions -= np.max(predictions, axis=1, keepdims=True)
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        loss = -np.log(probs[np.arange(batch_size), target_index.flatten()]).mean()
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # Compute softmax
    probs = softmax(predictions.copy())
    # Compute cross-entropy loss
    loss = cross_entropy_loss(probs, target_index)

    # Compute the gradient of predictions by loss value
    dprediction = probs.copy()
    if dprediction.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = dprediction.shape[0]
        dprediction[np.arange(batch_size), target_index.flatten()] -= 1
        dprediction /= batch_size

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, d_out):
        return d_out.copy() * (self.X > 0)

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        d_input = d_out.dot(self.W.value.T)
        self.W.grad += self.X.T.dot(d_out)
        self.B.grad += np.sum(d_out, axis=0, keepdims=True)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        X_padded = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        X_padded[:, self.padding:self.padding + height, self.padding:self.padding + width, :] = X
        self.X_cache = (X, X_padded)
        X_padded = X_padded[:, :, :, :, np.newaxis]
        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for y in range(out_height):
            for x in range(out_width):
                receptive_part = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, :]
                out[:, y, x, :] = np.sum(receptive_part * self.W.value, axis=(1, 2, 3)) + self.B.value
        return out

    def backward(self, d_out):
        X, X_padded = self.X_cache

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        X_grad = np.zeros_like(X_padded)

        for y in range(out_height):
            for x in range(out_width):
                reception_part = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(grad * reception_part, axis=0)
                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)
        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
        return X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                receptive_part = X[:, y * self.stride:y * self.stride + self.pool_size,
                                    x * self.stride:x * self.stride + self.pool_size, :]
                out[:, y, x, :] = np.max(receptive_part, axis=(1, 2))
        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        d_X = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                receptive_part = self.X[:, y * self.stride:y * self.stride + self.pool_size,
                                 x * self.stride:x * self.stride + self.pool_size, :]
                max_values = np.max(receptive_part, axis=(1, 2), keepdims=True)
                mask = (receptive_part == max_values)
                d_X[:, y * self.stride:y * self.stride + self.pool_size,
                    x * self.stride:x * self.stride + self.pool_size, :] += mask * d_out[:, y, x, np.newaxis, np.newaxis, :]
        return d_X

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
