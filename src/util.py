import matplotlib.pyplot as plt


# TODO: move file
def print_checks(train_data, valid_data, test_data, train_loader, valid_loader, test_loader):
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
        ax.axis('off')
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
    plt.axis('off')
    plt.show()


def loss_acc_diagram(train_losses, val_losses, train_accs, val_accs):
    """
    Displays a diagram showing the training and validation losses and accuracies over epochs.

    Parameters:
        train_losses (list): List of training losses for each epoch.

        val_losses (list): List of validation losses for each epoch.

        train_accs (list): List of training accuracies for each epoch.

        val_accs (list): List of validation accuracies for each epoch.

    Returns:
        None

    Displays:
        A diagram with two subplots showing the training and validation losses and accuracies.

    Note:
        - The 'train_losses' and 'val_losses' lists should contain loss values for each epoch.
        - The 'train_accs' and 'val_accs' lists should contain accuracy values for each epoch.
        - The lengths of all lists should be the same.
        - The losses and accuracies are plotted against the epoch number.
        - The first subplot displays the losses (y-axis) with 'train' and 'valid' lines in blue and green, respectively.
        - The second subplot displays the accuracies (y-axis) with 'train' and 'valid' lines in blue and green,
        respectively.
        - The x-axis represents the epoch number.
    """
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
    ax1.plot(train_losses, color='b', label='train')
    ax1.plot(val_losses, color='g', label='valid')
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(train_accs, color='b', label='train')
    ax2.plot(val_accs, color='g', label='valid')
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.show()
