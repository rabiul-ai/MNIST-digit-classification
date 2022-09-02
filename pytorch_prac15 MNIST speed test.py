#  MNIST Handwritten Digit Classification, MD RABIUL ISLAM, Date: 03.09.2022 at 01:39 AM

# steps: LDMTT
# 1. Library
# 2. Dataset
# 2. a. Data Download
# 2. b. Data Loading
# 2. c. Data Showing
# 3. Model:
# 3. a. Hyperparameter
# 3. b. Class
# 3. c. Loss & Optimizer
# 4. Training
# 5. Testing

# LDMTT
# 1. Library
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn

# 2. Data
# 2. a. Data Download
train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 2. b. Data Load
batch_size = 100
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# 2. c. Data Show
example = iter(train_loader)
example_data, example_lable = example.next()
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

# 3. Model
# 3. a. Hyperparameter
n_epoch = 2
input_size = 28 * 28
hidden_size = 500
n_classes = 10
learning_rate = 0.001


# 3. b. class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
print(device)
model = NeuralNet(input_size, hidden_size, n_classes).to(device)

# 3. c. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4. Training
n_total_step = len(train_loader)
for epoch in range(n_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        y_pred = model(images)
        loss = criterion(y_pred, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pring
        if (i + 1) % 100 == 0:
            print(f'Epoch= {epoch}/{n_epoch}, Steps= {i + 1}/{n_total_step}, Loss= {loss.item():.4f}')

# 5. Testing
n_correct = 0
n_samples = 0
for images, labels in test_loader:
    images = images.reshape(-1, 28 * 28).to(device)
    labels = labels.to(device)

    # forward pass
    y_pred = model(images)
    _, prediction = torch.max(y_pred.data, dim=1)

    n_correct = n_correct + (prediction == labels).sum().item()
    n_samples = n_samples + len(test_loader)

acc = (n_correct / n_samples) * 100
print(f'Accuracy of MNIST Handwritten digit classification data = {acc:.4f}')


# Alhamdulillah, fully unseen cody typing, within 21 minutes, just 1 error found and corrected by myself.
