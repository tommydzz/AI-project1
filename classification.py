import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split

epoch_losses = []
epoch_accuracies = []


class ClassificationNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, dropout_rate=0.2,
                 l2_lambda=0.001):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
        self.activations = []
        self.loss_history = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        np.random.seed(42)
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.m_bias = [np.zeros_like(b) for b in self.biases]
        self.v_bias = [np.zeros_like(b) for b in self.biases]
        self.t = 0
        self.dropout_rate = dropout_rate
        self.dropout_masks = []
        self.l2_lambda = l2_lambda

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def forward(self, x, is_training=True):
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            z = self.activations[-1].dot(self.weights[i]) + self.biases[i]
            a = self.relu(z)
            if is_training:
                # 应用Dropout
                a, mask = dropout_forward(a, self.dropout_rate)
                self.dropout_masks.append(mask)
            self.activations.append(a)

        # 最后一层不使用Dropout
        z = self.activations[-1].dot(self.weights[-1]) + self.biases[-1]
        output = self.softmax(z)
        self.activations.append(output)
        return output

    def backward(self, y_true):
        self.t += 1  # Increment time step for Adam
        y_pred = self.activations[-1]
        delta = y_pred - y_true
        d_weights = []
        d_biases = []

        # Calculate gradients for output layer
        d_weights.append(self.activations[-2].T.dot(delta))
        d_biases.append(np.sum(delta, axis=0, keepdims=True))

        # Propagate the error backwards
        for i in range(len(self.weights) - 1, 0, -1):
            delta = delta.dot(self.weights[i].T) * self.relu_derivative(self.activations[i])
            d_weight = self.activations[i - 1].T.dot(delta)
            d_bias = np.sum(delta, axis=0, keepdims=True)
            d_weights.append(d_weight)
            d_biases.append(d_bias)

        # Reverse the gradients lists as we appended from output to input
        d_weights.reverse()
        d_biases.reverse()

        # Update weights and biases using Adam
        for i in range(len(self.weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * d_weights[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (d_weights[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.m_bias[i] = self.beta1 * self.m_bias[i] + (1 - self.beta1) * d_biases[i]
            self.v_bias[i] = self.beta2 * self.v_bias[i] + (1 - self.beta2) * (d_biases[i] ** 2)
            m_hat_bias = self.m_bias[i] / (1 - self.beta1 ** self.t)
            v_hat_bias = self.v_bias[i] / (1 - self.beta2 ** self.t)
            self.biases[i] -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        epsilon = 1e-9  # 防止计算log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        log_likelihood = -np.log(y_pred[np.arange(m), y_true.argmax(axis=1)])
        l2_penalty = sum([(w ** 2).sum() for w in self.weights])  # L2
        loss = np.sum(log_likelihood) / m + self.l2_lambda * l2_penalty
        return loss

    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size):
        num_samples = x_train.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0
            batches_processed = 0

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                if start_idx == end_idx:
                    continue
                batch_x = x_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]

                y_pred = self.forward(batch_x, is_training=True)
                loss = self.compute_loss(y_pred, batch_y)
                self.loss_history.append(loss)

                epoch_loss += loss
                batches_processed += 1

                self.backward(batch_y)

            if batches_processed > 0:
                average_epoch_loss = epoch_loss / batches_processed
            else:
                average_epoch_loss = None

            epoch_losses.append(average_epoch_loss)

            accuracy = evaluate_model(self, x_test, y_test)
            epoch_accuracies.append(accuracy)
            print(f"Epoch {epoch}, Average Loss: {average_epoch_loss}, Accuracy: {accuracy}")


def load_and_process_images(base_dir, img_size=(28, 28)):
    images = []
    labels = []
    for label in range(1, 13):
        folder_path = os.path.join(base_dir, str(label))
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            with Image.open(img_path) as img:
                # 图片预处理
                img_resized = img.resize(img_size).convert('L')  # 调整大小并转换为灰度图
                img_array = np.array(img_resized) / 255.0  # 归一化
                images.append(img_array)
                labels.append(label - 1)

    return np.array(images), np.array(labels)


def dropout_forward(x, dropout_rate):
    if dropout_rate > 0:
        mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape) / (1 - dropout_rate)
        return x * mask, mask
    return x, None


def dropout_backward(dout, mask, dropout_rate):
    if dropout_rate > 0:
        return dout * mask
    return dout


def evaluate_model(network, x_test, y_test, is_training=False):
    predictions = network.forward(x_test, is_training)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy


def to_categorical(y, num_classes):
    # one hot
    return np.eye(num_classes)[y]


if __name__ == '__main__':
    Images, Labels = load_and_process_images(base_dir='.\\train')
    X = Images.reshape(Images.shape[0], -1)
    Y = to_categorical(Labels, 12)

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    input_size = 28 * 28
    output_size = 12
    hidden_layers = [150, 75]
    Layer_sizes = [input_size] + hidden_layers + [output_size]

    nn = ClassificationNeuralNetwork(Layer_sizes, learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                     dropout_rate=0.2)

    nn.train(X_train, Y_train, X_test, Y_test, epochs=200, batch_size=64)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epoch_losses, label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epoch_accuracies, label='Training Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Training Loss and Accuracy')
    plt.show()
