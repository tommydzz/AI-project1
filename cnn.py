import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from classification import load_and_process_images
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Assuming input images are 28x28
        self.fc2 = nn.Linear(128, 12)  # 12 classes

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output for the fully connected layer
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 保存模型状态字典
def save_model_state_dict(model, path):
    torch.save(model.state_dict(), path)


def load_model_state_dict(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # Evaluation mode


if __name__ == "__main__":
    images, labels = load_and_process_images(base_dir=".\\train")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # trans to PyTorch Tensors
    tensor_x_train = torch.Tensor(X_train).unsqueeze(1)
    tensor_y_train = torch.LongTensor(y_train)
    tensor_x_test = torch.Tensor(X_test).unsqueeze(1)
    tensor_y_test = torch.LongTensor(y_test)

    # create TensorDatasets
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    Model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Model.parameters(), lr=0.0001)

    # records
    epoch_losses = []
    epoch_accuracies = []

    # training
    num_epochs = 100
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = Model(inputs)
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
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")
    # torch.save(model.state_dict(), f'.\\model\\model_weights_epoch_{epoch + 1}.pth')

    # Graphing
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(epoch_losses, label="Training Loss", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(epoch_accuracies, label="Training Accuracy", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Accuracy")
    plt.show()
