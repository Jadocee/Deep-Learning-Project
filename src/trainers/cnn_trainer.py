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
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from models.resnet_model import ResNet18, Resblock
from utils.cnn_custom_dataset import CustomDataset
from utils.cnn_util import CNNUtils


class Trainer:
    """
    A class that handles training, validation, and testing of a model.

    Attributes:
    - train_data: The training data.
    - valid_data: The validation data.
    - test_data: The testing data.
    - train_losses: A list to store the training losses.
    - val_losses: A list to store the validation losses.
    - train_accs: A list to store the training accuracies.
    - val_accs: A list to store the validation accuracies.
    - train_loader: The data loader for training data.
    - test_loader: The data loader for testing data.
    - valid_loader: The data loader for validation data.
    - device: The device used for computations ("cuda" if available, else "cpu").
    """

    def __init__(self):
        """
        Initializes a new instance of the class.

        This method sets up the necessary attributes and variables for the class instance.

        Attributes:
        - train_data: The training data.
        - valid_data: The validation data.
        - test_data: The testing data.
        - train_losses: A list to store the training losses.
        - val_losses: A list to store the validation losses.
        - train_accs: A list to store the training accuracies.
        - val_accs: A list to store the validation accuracies.
        - train_loader: The data loader for training data.
        - test_loader: The data loader for testing data.
        - valid_loader: The data loader for validation data.
        - device: The device used for computations ("cuda" if available, else "cpu").
        """
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.train_loader, self.test_loader, self.valid_loader = None, None, None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_data(self, batch_size):
        """
        Prepares the data for training, validation, and testing.

        This method sets up the necessary data transformations, creates dataset instances,
        and initializes data loaders for training, validation, and testing.

        Args:
        - batch_size: The batch size to be used in the data loaders.

        Transforms:
        - train_transforms: A series of transformations applied to the training data, including random resized crop,
                            random horizontal flip, and conversion to tensor.
        - eval_transforms: A series of transformations applied to the evaluation data, including resize and conversion to tensor.

        Data Paths:
        - self.train_data: The training dataset.
        - self.valid_data: The validation dataset.
        - self.test_data: The testing dataset.

        Data Loaders:
        - self.train_loader: The data loader for training data.
        - self.valid_loader: The data loader for validation data.
        - self.test_loader: The data loader for testing data.

        Returns:
        None
        """
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(scale=(0.6, 1.0), size=(150, 150)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        eval_transforms = transforms.Compose(
            [transforms.Resize(size=(150, 150)), transforms.ToTensor()]
        )

        self.train_data = CustomDataset(
            "Deep-Learning-Project\data\intel_image_classification_dataset\seg_train\seg_train",
            transform=train_transforms,
        )

        self.valid_data = CustomDataset(
            "Deep-Learning-Project\data\intel_image_classification_dataset\seg_test\seg_test",
            transform=eval_transforms,
        )
        self.test_data = CustomDataset(
            "Deep-Learning-Project\data\mock_test",
            transform=eval_transforms,
        )

        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(
            dataset=self.valid_data, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            dataset=self.test_data, batch_size=batch_size, shuffle=False
        )

    def begin_training(self, str_model, width, num_epochs, learning_rate, count):
        """
        Begins the training process for the specified model.

        This method performs the training loop, updating the model's parameters based on the specified number of epochs,
        learning rate, and batch size. It also calculates and prints the training, validation, and test accuracies and losses.

        Args:
        - str_model: The name of the model to be used ("resnet18" or custom).
        - width: The width parameter for the custom model (applicable only if str_model is "custom").
        - num_epochs: The number of epochs to train the model.
        - learning_rate: The learning rate for the optimizer.
        - count: A counter for tracking the experiment number.

        Returns:
        None
        """
        if str_model == "resnet18":
            model = ResNet18(3, width, Resblock, outputs=1000)
        model = model.to(self.device)
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        loss_fn = CrossEntropyLoss()

        for epoch in range(num_epochs):
            epoch_loss, epoch_accuracy = 0, 0
            model.train()
            for i, (data, label) in enumerate(self.train_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                label = label.to(self.device)
                output = model(data)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(self.train_loader)
                epoch_loss += loss / len(self.train_loader)
            print(
                "Epoch: {}, train accuracy: {:.2f}%, train loss: {:.4f}".format(
                    epoch + 1, epoch_accuracy * 100, epoch_loss
                )
            )
            self.train_losses.append(epoch_loss.item())
            self.train_accs.append(epoch_accuracy.item())

            epoch_valid_accuracy, epoch_valid_loss = 0, 0
            model.eval()
            with torch.no_grad():
                for data, label in self.valid_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)
                    valid_output = model(data)
                    valid_loss = loss_fn(valid_output, label)
                    acc = (valid_output.argmax(dim=1) == label).float().mean()
                    epoch_valid_accuracy += acc / len(self.valid_loader)
                    epoch_valid_loss += valid_loss / len(self.valid_loader)
            print(
                "Epoch: {}, val accuracy: {:.2f}%, val loss: {:.4f}".format(
                    epoch + 1, epoch_valid_accuracy * 100, epoch_valid_loss
                )
            )
            self.val_losses.append(epoch_valid_loss.item())
            self.val_accs.append(epoch_valid_accuracy.item())

            y_true_test, y_pred_test = [], []
            test_accuracy, test_loss = 0, 0
            with torch.no_grad():
                for data, label in self.test_loader:
                    data = data.to(self.device)
                    label = label.to(self.device).long()
                    test_output_i = model(data)
                    test_loss_i = loss_fn(test_output_i, label)
                    acc = (test_output_i.argmax(dim=1) == label).float().mean()
                    test_accuracy += acc / len(self.test_loader)
                    test_loss += test_loss_i / len(self.test_loader)
                    y_true_test += label.tolist()
                    y_pred_test += test_output_i.argmax(dim=1).tolist()

        print("Final Test loss: {:.4f}".format(test_loss))
        print("Final Test accuracy: {:.2f}%".format(test_accuracy * 100))
        print(confusion_matrix(y_true_test, y_pred_test))
        print(classification_report(y_true_test, y_pred_test))
        hyper_params = [str_model, width, num_epochs, learning_rate]
        CNNUtils.generate_reports(y_true_test, y_pred_test, test_accuracy, hyper_params)
        CNNUtils.loss_acc_diagram(
            self.train_losses,
            self.val_losses,
            self.train_accs,
            self.val_accs,
            hyper_params,
            count,
        )


def menu_prompt(model, width, num_epochs, lr):
    """
    Prompts the user to select a model and begins the training process.

    This function takes in the user-selected model, width, number of epochs, and learning rate as parameters.
    It creates an instance of the Trainer class, prepares the data, and starts the training process using the selected model.

    Args:
    - model: The name of the model to be used ("resnet18" or custom).
    - width: The width parameter for the custom model (applicable only if model is "custom").
    - num_epochs: The number of epochs to train the model.
    - lr: The learning rate for the optimizer.

    Returns:
    None
    """
    count = 0
    test = Trainer()
    test.prepare_data(100)
    test.begin_training(model, width, num_epochs, lr, count)
