# File: train.py
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

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Dataset
from model import ResNet18, Resblock
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


class Train():
    '''
    Class for handling data loading, data splitting, training, and visualization of the Intel Image Classification dataset.

    Attributes:
        train_data (Dataset): The training dataset.

        valid_data (Dataset): The validation dataset.

        test_data (Dataset): The test dataset.

        device (str): The device to be used for computations ('cuda' if available, else 'cpu').
    '''

    def __init__(self):
        '''
        Initializes the Train class and sets the computation device.
        '''
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {}'.format(self.device))

    def prepare_data(self):
        '''
        Prepares the data by performing transformations, splitting the data into training, validation, and test sets,
        and loading them into DataLoader objects.
        '''
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
            dataset=self.train_data, batch_size=100, shuffle=True)  # TODO Update to take a set amount of batches vs arbitrary number
        self.valid_loader = DataLoader(
            dataset=self.valid_data, batch_size=100, shuffle=False)
        self.test_loader = DataLoader(
            dataset=self.test_data, batch_size=100, shuffle=False)
        # return self.train_loader, self.valid_loader, self.test_loader

    def begin_training(self, num_epochs):
        '''
        Trains a ResNet18 model on the prepared data for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs for training the model.
        '''
        model = ResNet18(3, Resblock, outputs=1000)
        model = model.to(self.device)
        optimizer = Adam(params=model.parameters(), lr=0.001)
        loss_fn = CrossEntropyLoss()
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
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
                epoch_accuracy += acc/len(self.train_loader)
                epoch_loss += loss/len(self.train_loader)
            print('Epoch: {}, train accuracy: {:.2f}%, train loss: {:.4f}'.format(
                epoch+1, epoch_accuracy*100, epoch_loss))
            train_losses.append(epoch_loss.item())
            train_accs.append(epoch_accuracy.item())
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
                    epoch_valid_accuracy += acc/len(self.valid_loader)
                    epoch_valid_loss += valid_loss/len(self.valid_loader)
            print('Epoch: {}, val accuracy: {:.2f}%, val loss: {:.4f}'.format(
                epoch+1, epoch_valid_accuracy*100, epoch_valid_loss))
            val_losses.append(epoch_valid_loss.item())
            val_accs.append(epoch_valid_accuracy.item())

    #-------------------- Util --------------------#

    def print_checks(self):
        '''
        Prints the size of the training, validation, and test datasets,
        the number of batches in each DataLoader, and the shape of a sample image.
        '''
        # Check our dataset sizes
        print("Train: {} examples".format(len(self.train_data)))
        print("Valid: {} examples".format(len(self.valid_data)))
        print("Test: {} examples".format(len(self.test_data)))

        # Check number of batches
        print("Train: {} batches".format(len(self.train_loader)))
        print("Valid: {} batches".format(len(self.valid_loader)))  # Should be 1
        print("Test: {} batches".format(len(self.test_loader)))  # Should be 1

        print(self.valid_data[0][0].shape)

    def diagram(self):
        '''
        Displays a 4x4 grid of sample images from the training data.
        '''
        fig = plt.figure()
        fig.set_figheight(12)
        fig.set_figwidth(12)
        for idx in range(16):
            ax = fig.add_subplot(4, 4, idx+1)
            ax.axis('off')
            if self.train_data[idx][1] == 0:
                ax.set_title("buildings")
            elif self.train_data[idx][1] == 1:
                ax.set_title("forest")
            elif self.train_data[idx][1] == 2:
                ax.set_title("glacier")
            elif self.train_data[idx][1] == 3:
                ax.set_title("mountain")
            elif self.train_data[idx][1] == 4:
                ax.set_title("sea")
            elif self.train_data[idx][1] == 5:
                ax.set_title("street")
            plt.imshow(self.train_data[idx][0].permute(1, 2, 0))
        plt.axis('off')
        plt.show()
