# Author: Thomas Bandy
#
# This script contains utility functions for analyzing and visualizing datasets and model performance using matplotlib.
#
# Functions:
# - print_checks(train_data, valid_data, test_data, train_loader, valid_loader, test_loader): Prints various checks and information about the datasets and loaders.
# - picture_diagram(data, grid_size): Displays a picture diagram using matplotlib with titles corresponding to different classes.
# - loss_acc_diagram(train_losses, val_losses, train_accs, val_accs): Displays a diagram showing the training and validation losses and accuracies over epochs.


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from openpyxl import load_workbook


@staticmethod
def print_checks(
    train_data, valid_data, test_data, train_loader, valid_loader, test_loader
):
    """
    Prints various checks and information about the datasets and loaders.

    Args:
        train_data (Dataset): Training dataset.
        valid_data (Dataset): Validation dataset.
        test_data (Dataset): Test dataset.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.

    Prints:
        - Number of examples in each dataset.
        - Number of batches in each loader.
        - Shape of the first element in the valid_data dataset.

    Note:
        This function assumes that the datasets are compatible with the data loaders.
    """
    # Check our dataset sizes
    print("Train: {} examples".format(len(train_data)))
    print("Valid: {} examples".format(len(valid_data)))
    print("Test: {} examples".format(len(test_data)))

    # Check number of batches
    print("Train: {} batches".format(len(train_loader)))
    print("Valid: {} batches".format(len(valid_loader)))
    print("Test: {} batches".format(len(test_loader)))

    print(valid_data[0][0].shape)


@staticmethod
def picture_diagram(data, grid_size):
    """
    Displays a picture diagram using matplotlib with titles corresponding to different classes.

    Parameters:
        data (list): A list of tuples containing images and their corresponding labels.

        grid_size (int): The size of the grid to arrange the images.

    Returns:
        None

    Displays:
        A picture diagram with images and their class titles.

    Note:
        - The 'data' parameter should contain tuples in the format (image, label).
        - The 'label' values should correspond to the following classes:

            0: buildings

            1: forest

            2: glacier

            3: mountain

            4: sea

            5: street
        - The 'image' should be a tensor with dimensions (channels, height, width).
    """
    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    for idx in range(16):
        ax = fig.add_subplot(grid_size, grid_size, idx + 1)
        ax.axis("off")
        if data[idx][1] == 0:
            ax.set_title("buildings")
        elif data[idx][1] == 1:
            ax.set_title("forest")
        elif data[idx][1] == 2:
            ax.set_title("glacier")
        elif data[idx][1] == 3:
            ax.set_title("mountain")
        elif data[idx][1] == 4:
            ax.set_title("sea")
        elif data[idx][1] == 5:
            ax.set_title("street")
        plt.imshow(data[idx][0].permute(1, 2, 0))
    plt.axis("off")
    plt.show()


@staticmethod
def loss_acc_diagram(
    train_losses, val_losses, train_accs, val_accs, hyper_params, count
):
    """
    Generates and saves a loss-accuracy diagram.

    This static method plots the training and validation losses and accuracies over epochs
    and saves the resulting diagram as an image file.

    Args:
    - train_losses: A list of training losses.
    - val_losses: A list of validation losses.
    - train_accs: A list of training accuracies.
    - val_accs: A list of validation accuracies.
    - hyper_params: A list of hyperparameters used for the model training.
    - count: The count used to distinguish the output image files.

    Returns:
    None
    """
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
    ax1.plot(train_losses, color="b", label="train")
    ax1.plot(val_losses, color="g", label="valid")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(train_accs, color="b", label="train")
    ax2.plot(val_accs, color="g", label="valid")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    filename = f"{hyper_params[0]}_w:{hyper_params[1]}_e:{hyper_params[2]}_lr:{hyper_params[3]}"
    plt.savefig(f"test{count}.png")


@staticmethod
def generate_reports(y_true, y_pred, accuracy, hyper_params, path="resnet_results.txt"):
    """
    Generates and saves reports including confusion matrix, classification report, and accuracy.

    This static method calculates the confusion matrix and classification report based on the true and predicted labels,
    and saves the results along with the accuracy in a text file.

    Args:
    - y_true: The true labels.
    - y_pred: The predicted labels.
    - accuracy: The accuracy of the model.
    - hyper_params: A list of hyperparameters used for the model training.
    - path: The file path to save the reports (default: "resnet_results.txt").

    Returns:
    None
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    with open(path, "a") as f:
        f.write(
            f"--------------- {hyper_params[0]}, w:{hyper_params[1]}, e:{hyper_params[2]}, lr:{hyper_params[3]} ---------------\n"
            f"Confusion Matrix - \n {conf_mat} \n \n Classification Report - \n {class_report} \n \n"
            f"Accuracy: {accuracy}\n \n"
        )
