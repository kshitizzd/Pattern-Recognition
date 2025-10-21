import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class CNN(nn.Module):
    """Convolutional Neural Network"""

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def train_cnn(X_train, y_train, X_test, y_test, epochs=20, lr=0.001, batch_size=64):
    """Train CNN"""
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track training history
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        # Calculate average loss and accuracy for this epoch
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {accuracy:.4f}")

    return model, accuracy.item(), train_losses, train_accuracies


