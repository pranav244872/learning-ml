import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image

resize_transforms = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

in_folders = ['./Rice/Ipsala/', './Rice/Arborio/', './Rice/Basmati/', './Rice/Jasmine/','./Rice/Karacadag/' ]

# Path to the root directory containing subdirectories for each class
root_dir = './Rice/'

# Create the dataset using ImageFolder
dataset = datasets.ImageFolder(root=root_dir, transform=resize_transforms)

# Define batch size
batch_size = 256 

# Determine the sizes of the training and test datasets
total_size = len(dataset)
test_size = int(0.2 * total_size)  # 20% for testing
train_size = total_size - test_size  # Remaining 80% for training

# Split the dataset into training and test datasets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=4)
