import numpy as np


class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss_history = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            z = self.activations[-1].dot(self.weights[i]) + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)
        z = self.activations[-1].dot(self.weights[-1]) + self.biases[-1]
        self.activations.append(z)
        return z

    def backward(self, x, y):
        # Forward pass
        self.forward(x)
        # Backward pass
        d_activations = [2.0 * (self.activations[-1] - y) / y.size]
        for i in reversed(range(len(self.weights))):
            d_weights = self.activations[i].T.dot(d_activations[-1])
            d_biases = np.sum(d_activations[-1], axis=0, keepdims=True)
            if i > 0:
                d_activations.append(
                    d_activations[-1].dot(self.weights[i].T) * self.relu_derivative(self.activations[i]))
            # Update weights and biases
            self.weights[i] -= self.learning_rate * d_weights
            self.biases[i] -= self.learning_rate * d_biases
        d_activations.reverse()

    def loss(self, x, y):
        return np.mean((self.forward(x) - y) ** 2)

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            loss = self.loss(x, y)
            self.loss_history.append(loss)
            self.backward(x, y)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
