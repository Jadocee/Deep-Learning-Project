# File: dataset.py
#
# Author: Thomas Bandy
#
# This file contains the implementation of the Dataset class for loading images and labels from a directory.
#
# All docstrings were generated by ChatGPT.
#
# Date: May 12, 2023

import glob
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# TODO: move file
class Dataset(Dataset):  # TODO: Redeclared 'Dataset' defined above without usage; Rename the element.
    """
    Custom dataset class for loading images and labels from a directory.

        Args:
            data_dir (str): Directory path containing the images and labels.
            transform (callable, optional): Optional transform to be applied to the images. Defaults to None.

        Attributes:
            image_paths (list): List of paths to the image files.
            labels (numpy.ndarray): Array of labels corresponding to the images.
            transform (callable): Transform to be applied to the images.
            n (int): Number of samples in the dataset.

        Methods:
            __len__(): Returns the number of samples in the dataset.
            __getitem__(idx): Retrieves the image and its corresponding label at the given index.
    """

    def __init__(self, data_dir, transform=None):
        # TODO: add docstring; I moved the docstring that was originally here to the class docstring
        #  (assuming that it was intended to be there).
        if os.path.exists(os.path.join(data_dir + "\\buildings")):
            folders = ['buildings', 'forest',
                       'glacier', 'mountain', 'sea', 'street']
            self.image_paths = []
            for folder in folders:
                updated_dir = os.path.join(data_dir + f'\\{folder}')
                self.image_paths += glob.glob(
                    os.path.join(updated_dir, '*.jpg'))
        else:
            self.image_paths = glob.glob(
                os.path.join(data_dir, '*.jpg'))

        self.image_paths.sort()
        labels_path = os.path.join(data_dir, "labels.csv")
        self.labels = pd.read_csv(labels_path, header=None).to_numpy()[
                      :, 1] if os.path.isfile(labels_path) else None
        self.transform = transform
        self.n = len(self.image_paths)

    def __len__(self):
        """ Returns the number of samples in the dataset.

            Returns:
                int: Number of samples in the dataset.
        """
        return self.n

    def __getitem__(self, idx):
        """ Retrieves the image and its corresponding label at the given index.

            Args:
                idx (int): Index of the sample to retrieve.

            Returns:
                tuple: Tuple containing the transformed image and its label.
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        if self.labels is None:
            if img_path.split('\\')[-2] == 'buildings':
                label = 0
            elif img_path.split('\\')[-2] == 'forest':
                label = 1
            elif img_path.split('\\')[-2] == 'glacier':
                label = 2
            elif img_path.split('\\')[-2] == 'mountain':
                label = 3
            elif img_path.split('\\')[-2] == 'sea':
                label = 4
            else:
                label = 5
        else:
            label = self.labels[idx]
        return img_transformed, label
