import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from network import SimpleNeuralNetwork
from sklearn.model_selection import train_test_split


class ClassificationNeuralNetwork(SimpleNeuralNetwork):
    def __init__(self, layer_sizes, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layer_sizes, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.m_bias = [np.zeros_like(b) for b in self.biases]
        self.v_bias = [np.zeros_like(b) for b in self.biases]
        self.t = 0
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def forward(self, x):
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            z = self.activations[-1].dot(self.weights[i]) + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)
        z = self.activations[-1].dot(self.weights[-1]) + self.biases[-1]
        output = self.softmax(z)
        self.activations.append(output)
        return output

    def backward(self, x, y_true):
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
        epsilon = 1e-9  # 小常数，防止计算对数0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 将预测值限制在[epsilon, 1-epsilon]区间内
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss


    def train(self, x, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)
            self.backward(x, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


def load_and_process_images(base_dir, img_size=(28, 28)):
    images = []
    labels = []
    for label in range(1, 13):  # 对于每一个类别
        folder_path = os.path.join(base_dir, str(label))
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            with Image.open(img_path) as img:
                # 图片预处理
                img_resized = img.resize(img_size).convert('L')  # 调整大小并转换为灰度图
                img_array = np.array(img_resized) / 255.0  # 归一化
                images.append(img_array)
                labels.append(label - 1)  # 标签从0开始

    return np.array(images), np.array(labels)


def evaluate_model(network, X_test, Y_test):
    predictions = network.forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy


def to_categorical(y, num_classes):
    """ 将标签向量转换为one-hot编码的矩阵 """
    return np.eye(num_classes)[y]


if __name__ == '__main__':
    images, labels = load_and_process_images(base_dir='.\\train')
    # 假设`images`和`labels`是从之前的load_and_process_images函数中获得的
    X = images.reshape(images.shape[0], -1)  # 展平图片数据
    Y = to_categorical(labels, 12)  # 转换标签为one-hot编码，12分类任务

    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    input_size = 28 * 28  # 图片大小为28x28
    output_size = 12  # 12个类别
    hidden_layers = [128, 64]  # 隐藏层的大小
    layer_sizes = [input_size] + hidden_layers + [output_size]

    # 初始化网络
    nn = ClassificationNeuralNetwork(layer_sizes, learning_rate=0.0001)

    # 训练网络
    nn.train(X_train, Y_train, epochs=5000)
    plt.plot(nn.loss_history)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # 计算测试集上的准确率
    accuracy = evaluate_model(nn, X_test, Y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
