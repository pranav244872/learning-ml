import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as datasets
from torch.utils.data import DataLoader
from neural_net import SimpleCNN
from data_processing import train_loader, test_loader, train_dataset
import math
# Hyper Parameters
criterion = nn.CrossEntropyLoss()
model = SimpleCNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_losses = []
val_losses = []

epochs = 2
total_samples = len(train_dataset)
n_iterations = math.ceil(total_samples/256)

# Training

for epoch in range(epochs):

    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        print(f'epoch {epoch+1}/{epochs}, step {i+1} / {n_iterations} ')

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'loss = {loss}')
        train_losses.append(loss.item())

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                print(f'Validation Loss = {loss.item()}')
                val_losses.append(loss.item())
                break

# nv1 = nn.Conv2d(in_channels = 3, out_channels= 16, kernel_size=3, stride=1, padding=1)
# nv2 = nn.Conv2d(in_channels = 16, out_channels= 32, kernel_size=3, stride=1, padding=1)
# nv3 = nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size=3, stride=1, padding=1)
#
# pool = nn.MaxPool2d(2, 2)
# for batch_idx, (inputs, labels) in enumerate(train_loader):
#     print(f'Batch {batch_idx + 1}:')
#     print(f'Inputs shape: {inputs.shape}')
#     print(f'Labels shape: {labels.shape}')
#     x = conv1(inputs)
#     print(f'x_shape = {x.shape}')
#     x = pool(x)
#     print(f'x_shape = {x.shape}')
#     x = conv2(x)
#     print(f'x_shape = {x.shape}')
#     x = pool(x)
#     print(f'x_shape = {x.shape}')
#     x = conv3(x)
#     print(f'x_shape = {x.shape}')
#     x = pool(x)
#     print(f'x_shape = {x.shape}')
#     # Process the batch
#     break
