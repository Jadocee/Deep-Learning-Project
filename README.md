# Deep-Learning-Project

Description of the project.

## Table of Contents

- [Train](#train)
- [Dataset](#dataset)
- [ResNet18](#resnet18)
- [Residual Block](#residual-block)

## Train

File: train.py

Author: Thomas Bandy

This file contains the implementation of the Train class, which handles training setup and data handling.

The Train class provides methods for splitting the dataset into training, validation, and test sets,
loading the data into DataLoader objects, printing information about the dataset and DataLoader,
and displaying a diagram of sample images from the training data.

All docstrings were written by ChatGPT.

Date: May 12, 2023

### Usage

```python
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Dataset
import matplotlib.pyplot as plt

# Create an instance of the Train class
train = Train()

# Split the dataset into training, validation, and test sets
train.training_split()

# Load the training, validation, and test data into DataLoader objects
train_loader, valid_loader, test_loader = train.load_data()

# Print information about the dataset and DataLoader
train.print_checks()

# Display a diagram of sample images from the training data
train.diagram()
