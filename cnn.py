import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from classification import load_and_process_images
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 权重和偏置的初始化
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels * kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # 获取输入的尺寸信息
        batch_size, _, height, width = x.shape
        height_out = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        width_out = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 添加填充
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # 展开输入以形成窗口
        x_unfolded = F.unfold(x_padded, kernel_size=self.kernel_size, stride=self.stride, padding=0)

        # 进行卷积：窗口和权重的矩阵乘法
        out_unfolded = self.weight.matmul(x_unfolded) + self.bias.view(-1, 1)

        # 转换输出格式
        out = out_unfolded.view(batch_size, self.out_channels, height_out, width_out)
        return out
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = CustomConv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = CustomConv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Assuming input images are 28x28
        self.fc2 = nn.Linear(128, 12)  # 12 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # 假设 images 和 labels 已经是加载好的 NumPy 数组
    images, labels = load_and_process_images(base_dir='.\\train')
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 转换为 PyTorch Tensors
    tensor_x_train = torch.Tensor(X_train).unsqueeze(1)  # 增加一个通道维度
    tensor_y_train = torch.LongTensor(y_train)
    tensor_x_test = torch.Tensor(X_test).unsqueeze(1)
    tensor_y_test = torch.LongTensor(y_test)

    # 创建 TensorDatasets
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 准备记录损失和准确率
    epoch_losses = []
    epoch_accuracies = []

    # 训练网络
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')

    # 绘制损失和准确率曲线
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epoch_losses, label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 实例化一个共享相同x轴的第二个坐标轴
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # 我们已经为这个坐标轴设置了标签
    ax2.plot(epoch_accuracies, label='Training Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # 以防标签被剪辑
    plt.title('Training Loss and Accuracy')
    plt.show()
