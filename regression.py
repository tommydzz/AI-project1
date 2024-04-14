import numpy as np
import matplotlib.pyplot as plt
from network import SimpleNeuralNetwork

if __name__ == '__main__':
    # 数据准备
    x = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)  # 形状为(200, 1)
    y = np.sin(x)

    # 初始化并训练神经网络
    nn = SimpleNeuralNetwork([1, 10, 1], 0.01)
    nn.train(x, y, epochs=10000)

    # 测试训练好的模型
    y_pred = nn.forward(x)
    plt.plot(x, y, label='True')
    plt.plot(x, y_pred, label='Predicted')
    plt.legend()
    plt.show()

    plt.semilogy(nn.loss_history)
    plt.plot(nn.loss_history)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()