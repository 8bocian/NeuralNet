import numpy as np
import random
import tensorflow as tf


def get_random():
    return (random.random() - 0.5) * 2


class NeuralNet:
    def __init__(self, layers_conf=(28*28, 16, 16, 10), l_r=0.001):
        self.layers = [
            Layer(input_size=layers_conf[idx], output_size=layers_conf[idx+1]) for idx in range(len(layers_conf)-1)
        ]
        self.l_r = l_r

    def predict(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer.forward_pass(outputs)
            outputs = self.sigmoid(outputs)
        return outputs

    def train(self, X, Y):
        for X_, y_true in zip(X, Y):
            activated_outputs = X_
            outputs_storage = []
            for layer in self.layers:
                outputs = layer.forward_pass(activated_outputs)
                activated_outputs = self.sigmoid(outputs)
                outputs_storage.append((outputs, activated_outputs))
            y_pred = activated_outputs
            layer_error = 2 * (y_pred - y_true) * self.sigmoid_derivative(outputs_storage[-1][0])

            for layer_idx in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_idx]
                weights_grad = np.outer(layer_error, outputs_storage[layer_idx - 1][1] if layer_idx > 0 else X_)
                bias_grad = layer_error

                layer.weights -= self.l_r * weights_grad
                layer.biases -= self.l_r * bias_grad

                if layer_idx > 0:
                    layer_error = np.dot(layer.weights.T, layer_error) * self.sigmoid_derivative(outputs_storage[layer_idx - 1][0])

            # dC/dW(L) = dC/dA(L) * dA(L)/dZ(L) * dZ(L)/dW(L) or dZ(L)/dB(L) or dZ/A(L-1)
            # 2(a(L) - y) * sigm(z(L))' * a(L-1) or w(L) or 1

    def sigmoid_derivative(self, x):
        x_ = self.sigmoid(x)
        return x_ * (1 - x_)

    def cost_function(self, y_true, y_pred):
        return np.sum((y_pred-y_true)**2, dtype=np.float64)

    def ReLU(self, x):
        return np.maximum(x, 0)

    def sigmoid(self, x):
        # if not isinstance(x, np.ndarray):
        #     x = np.array(x)
        return 1/(1+np.exp(-x))

    def __len__(self):
        return sum([len(layer) for layer in self.layers])

class Layer:
    def __init__(self, input_size, output_size):
        self.weights_size = input_size * output_size
        self.biases_size = output_size


        self.weights = np.array([
            [get_random() for i in range(input_size)] for i in range(output_size)
        ])
        self.biases = np.array([
            get_random() for i in range(output_size)
        ])

    def __len__(self):
        return self.weights_size + self.biases_size

    def forward_pass(self, inputs):
        dot = np.dot(self.weights, inputs)
        result = dot + self.biases
        return result


# NN = NeuralNet()
# X = [get_random() for i in range(28*28)]
# NN.predict(X)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

NN = NeuralNet()

epochs = 100
for epoch in range(epochs):
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]

    NN.train(x_train, y_train)

    predictions = np.array([NN.predict(x) for x in x_test])
    predictions = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_true)
    print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")