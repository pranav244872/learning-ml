import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as datasets
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(16 * 32 * 32, 15)  # Adjust according to the image size
        self.bn3 = nn.BatchNorm1d(15)
        self.fc2 = nn.Linear(15, 5)          # Number of output classes
    
    def forward(self, x):
        x = self.pool(torch.tanh(self.bn1(self.conv1(x))))
        x = self.pool(torch.tanh(self.bn2(self.conv2(x))))
        
        x = x.view(-1, 16 * 32 * 32)  # Flatten the tensor
        
        x = torch.tanh((self.fc1(x)))
        x = self.bn3(x)

        x = self.fc2(x) 

        return x
