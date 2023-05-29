import warnings
from os import getenv, environ
from os.path import join, exists
from typing import Dict, Final

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Resize

from custom_datasets.cnn_custom_dataset import CNNCustomDataset
from utils.definitions import DATA_DIR


class DatasetLoader:
    """
    A static utility class responsible for loading custom_datasets. This class should not be instantiated.

    Attributes:
        TWEET_TOPIC_SINGLE (Final[str]): The name of the Tweet Topic Single dataset.
        TWEET_TOPIC_SINGLE_TRAIN_SPLIT (Final[str]): The name of the train split of the Tweet Topic Single dataset that
            is used for training and validating the models in this project.
        TWEET_TOPIC_SINGLE_TEST_SPLIT (Final[str]): The name of the test split of the Tweet Topic Single dataset that is
            used for testing the models in this project.
    """

    TWEET_TOPIC_SINGLE: Final[str] = "cardiffnlp/tweet_topic_single"
    TWEET_TOPIC_SINGLE_TRAIN_SPLIT: Final[str] = "train_coling2022"
    TWEET_TOPIC_SINGLE_TEST_SPLIT: Final[str] = "test_coling2022"
    INTEL_IMAGE_CLASSIFICATION: Final[str] = "puneet6060/intel-image-classification"

    def __init__(self) -> None:
        """
        Raise an exception if an attempt to instantiate the DatasetLoader class is made.

        Raises:
            Exception: An error indicating that this class should not be instantiated.
        """
        raise Exception("This class should not be instantiated.")

    @staticmethod
    def get_intel_image_classification_dataset() -> Dict[str, CNNCustomDataset]:
        """
        Load the Intel Image Classification dataset.

        Returns:
            Dict[str, CNNCustomDataset]: A dictionary containing the training, validation, and testing sets.

        Notes:
            - If the dataset is not found in the data directory, it will be downloaded from Kaggle; the Kaggle API
              credentials must be provided in the .env file.

        References:
            - https://www.kaggle.com/datasets/puneet6060/intel-image-classification
            - https://www.kaggle.com/docs/api
        """

        dataset_path: str = join(DATA_DIR, "intel_image_classification_dataset")
        if not exists(dataset_path):
            kaggle_username: str = str(getenv("KAGGLE_USERNAME"))
            kaggle_key: str = str(getenv("KAGGLE_KEY"))
            environ["KAGGLE_USERNAME"] = kaggle_username
            environ["KAGGLE_KEY"] = kaggle_key
            from kaggle.api.kaggle_api_extended import KaggleApi
            kaggle_api: KaggleApi = KaggleApi()
            kaggle_api.authenticate()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kaggle_api.dataset_download_files(dataset=DatasetLoader.INTEL_IMAGE_CLASSIFICATION,
                                                  path=dataset_path,
                                                  force=False,
                                                  unzip=True)
            environ.clear()

        train_transforms: Compose = Compose([
            RandomResizedCrop(scale=(0.6, 1.0), size=(150, 150)),
            RandomHorizontalFlip(),
            ToTensor()
        ])
        eval_transforms: Compose = Compose([
            Resize(size=(150, 150)),
            ToTensor()
        ])

        train_dataset: CNNCustomDataset = CNNCustomDataset(data_dir=join(dataset_path, "seg_train", "seg_train"),
                                                           transform=train_transforms)
        test_dataset: CNNCustomDataset = CNNCustomDataset(data_dir=join(dataset_path, "seg_test", "seg_test"),
                                                           transform=eval_transforms)
        valid_dataset: CNNCustomDataset = CNNCustomDataset(data_dir=join(dataset_path, "seg_pred", "seg_pred"),
                                                          transform=eval_transforms)

        return {
            "train": train_dataset,
            "validation": valid_dataset,
            "test": test_dataset
        }

    @staticmethod
    def get_tweet_topic_single_dataset(test_size: float = 0.2) -> DatasetDict[str, Dataset]:
        """
        Load and split the Tweet Topic Single dataset into training, validation, and testing sets.

        Args:
            test_size (float, optional): The proportion of the dataset to include in the testing set. Defaults to 0.2
                (20% of the dataset).

        Returns:
            DatasetDict[str, Dataset]: A dictionary containing the training, validation, and testing sets.

        References:
            - https://huggingface.co/datasets/cardiffnlp/tweet_topic_single
        """
        output_dir: Final[str] = join(DATA_DIR, "tweet_topic_single")
        storage_dir: Final[str] = join(output_dir, ".storage")
        cache_dir: Final[str] = join(output_dir, ".cache")

        dataset_dict: DatasetDict
        try:
            dataset_dict = load_from_disk(storage_dir)
        except FileNotFoundError:
            train_data: Dataset = load_dataset(
                path=DatasetLoader.TWEET_TOPIC_SINGLE,
                split=DatasetLoader.TWEET_TOPIC_SINGLE_TRAIN_SPLIT,
                cache_dir=None,
                keep_in_memory=False
            )
            test_data: Dataset = load_dataset(
                path=DatasetLoader.TWEET_TOPIC_SINGLE,
                split=DatasetLoader.TWEET_TOPIC_SINGLE_TEST_SPLIT,
                cache_dir=None,
                keep_in_memory=False
            )
            train_valid_split: DatasetDict = train_data.train_test_split(test_size=test_size)
            train_data: Dataset = train_valid_split["train"]
            valid_data: Dataset = train_valid_split["test"]
            dataset_dict = DatasetDict({"train": train_data, "validation": valid_data, "test": test_data})
            dataset_dict.save_to_disk(storage_dir)
        return dataset_dict
