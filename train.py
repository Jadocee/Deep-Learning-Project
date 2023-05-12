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
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Dataset


class Train():
    ''' 
        Class for training setup and data handling.

        Methods:
            training_split(): Splits the dataset into training, validation, and test sets.

            load_data(): Loads the training, validation, and test data into DataLoader objects.

            print_checks(): Prints information about the dataset and DataLoader.

            diagram(): Displays a diagram of sample images from the training data.
    '''

    def __init__(self):
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def training_split(self):
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

    def load_data(self):
        '''
            Splits the dataset into training, validation, and test sets and applies transformations to each set.
        '''
        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=100, shuffle=True)  # TODO Update to take a set amount of batches vs arbitrary number
        self.valid_loader = DataLoader(
            dataset=self.valid_data, batch_size=len(self.valid_data), shuffle=False)
        self.test_loader = DataLoader(
            dataset=self.test_data, batch_size=len(self.test_data), shuffle=False)
        return self.train_loader, self.valid_loader, self.test_loader

    def print_checks(self):
        '''
            Loads the training, validation, and test data into DataLoader objects.

            Returns:
                tuple: Tuple containing the training, validation, and test DataLoader objects.
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
            Displays a diagram of sample images from the training data.
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
