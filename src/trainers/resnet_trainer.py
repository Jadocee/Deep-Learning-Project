# File: resnet_trainer.py
#
# Author: Thomas Bandy
#
# This file contains the implementation of the Train class, which handles training setup and data handling.
#
# The Train class provides methods for splitting the dataset into training, validation, and test sets,
# loading the data into DataLoader objects, printing information about the dataset and DataLoader,
# and displaying a diagram of sample images from the training data.
#
# All docstrings were written by ChatGPT.
#
# Date: May 12, 2023

import torch
from Models.resnet_model import ResNet18, Resblock
from Util.dataset import Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms


class ResNetTrainer:
    """
    Class for handling data loading, data splitting, training, and visualization of the Intel Image Classification dataset.

    Attributes:
        train_data (Dataset): The training dataset.

        valid_data (Dataset): The validation dataset.

        test_data (Dataset): The test dataset.

        device (str): The device to be used for computations ('cuda' if available, else 'cpu').
    """

    def __init__(self):
        """
        Initializes the Train class and sets the computation device.
        """
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {}'.format(self.device))

    def prepare_data(self):
        """
        Prepares the data by performing transformations, splitting the data into training, validation, and test sets,
        and loading them into DataLoader objects.
        """
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(scale=(0.6, 1.0), size=(150, 150)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        eval_transforms = transforms.Compose([
            transforms.Resize(size=(150, 150)),
            transforms.ToTensor()
        ])

        # Create our dataset splits
        self.train_data = Dataset(
            "data\intel_image_classification_dataset\seg_train\seg_train", transform=train_transforms)
        self.valid_data = Dataset(
            "data\intel_image_classification_dataset\seg_pred\seg_pred", transform=eval_transforms)
        self.test_data = Dataset(
            "data\intel_image_classification_dataset\seg_test\seg_test", transform=eval_transforms)

        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=100,
            shuffle=True)  # TODO Update to take a set amount of batches vs arbitrary number
        self.valid_loader = DataLoader(
            dataset=self.valid_data, batch_size=100, shuffle=False)
        self.test_loader = DataLoader(
            dataset=self.test_data, batch_size=100, shuffle=False)
        # return self.train_loader, self.valid_loader, self.test_loader

    def begin_training(self, width, num_epochs, learning_rate):
        """
        Trains a ResNet18 model on the prepared data for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs for training the model.
        """
        model = ResNet18(3, width, Resblock, outputs=1000)
        model = model.to(self.device)
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        loss_fn = CrossEntropyLoss()
        # TODO: Instance attributes train_losses, val_losses, train_accs, val_accs
        #  defined outside __init__ method; should be defined inside __init__
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        # Train for 10 epochs
        for epoch in range(num_epochs):
            # Training
            # Track epoch loss and accuracy
            epoch_loss, epoch_accuracy = 0, 0
            # Switch model to training (affects batch norm and dropout)
            model.train()
            # Iterate through batches
            for i, (data, label) in enumerate(self.train_loader):
                # Reset gradients
                optimizer.zero_grad()
                # Move data to the used device
                data = data.to(self.device)
                label = label.to(self.device)
                # Forward pass
                output = model(data)
                loss = loss_fn(output, label)
                # Backward pass
                loss.backward()
                # Adjust weights
                optimizer.step()
                # Compute metrics
                acc = ((output.argmax(dim=1) == label).float().mean())
                epoch_accuracy += acc / len(self.train_loader)
                epoch_loss += loss / len(self.train_loader)
            print('Epoch: {}, train accuracy: {:.2f}%, train loss: {:.4f}'.format(
                epoch + 1, epoch_accuracy * 100, epoch_loss))
            self.train_losses.append(epoch_loss.item())
            self.train_accs.append(epoch_accuracy.item())
            # Evaluation
            # Track epoch loss and accuracy
            epoch_valid_accuracy, epoch_valid_loss = 0, 0
            # Switch model to evaluation (affects batch norm and dropout)
            model.eval()
            # Disable gradients
            with torch.no_grad():
                # Iterate through batches
                for data, label in self.valid_loader:
                    # Move data to the used device
                    data = data.to(self.device)
                    label = label.to(self.device)
                    # Forward pass
                    valid_output = model(data)
                    valid_loss = loss_fn(valid_output, label)
                    # Compute metrics
                    acc = ((valid_output.argmax(dim=1) == label).float().mean())
                    epoch_valid_accuracy += acc / len(self.valid_loader)
                    epoch_valid_loss += valid_loss / len(self.valid_loader)
            print('Epoch: {}, val accuracy: {:.2f}%, val loss: {:.4f}'.format(
                epoch + 1, epoch_valid_accuracy * 100, epoch_valid_loss))
            self.val_losses.append(epoch_valid_loss.item())
            self.val_accs.append(epoch_valid_accuracy.item())

            test_accuracy, test_loss = 0, 0
            with torch.no_grad():
                # Iterate through batches
                for data, label in self.test_loader:
                    # Move data to the used device
                    data = data.to(self.device)
                    label = label.to(self.device)
                    # Forward pass
                    test_output_i = model(data)
                    test_loss_i = loss_fn(test_output_i, label)
                    # Compute metrics
                    acc = ((test_output_i.argmax(dim=1) == label).float().mean())
                    test_accuracy += acc / len(self.test_loader)
                    test_loss += test_loss_i / len(self.test_loader)

        print("Final Test loss: {:.4f}".format(test_loss))
        print("Final Test accuracy: {:.2f}%".format(test_accuracy * 100))
