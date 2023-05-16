from typing import List, Tuple

from datasets import load_dataset, get_dataset_split_names, Dataset, DatasetDict


class DatasetLoader:
    """
    A static utility class responsible for loading datasets. This class should not be instantiated.
    """

    TWEET_TOPIC_SINGLE: str = "cardiffnlp/tweet_topic_single"
    """
    The name of the Tweet Topic Single dataset.
    """

    TWEET_TOPIC_SINGLE_TRAIN_SPLIT: str = "train_coling2022"
    """
    The name of the train split of the Tweet Topic Single dataset.
    """

    TWEET_TOPIC_SINGLE_TEST_SPLIT: str = "test_coling2022"
    """
    The name of the test split of the Tweet Topic Single dataset.
    """

    def __init__(self) -> None:
        """
        Raise an exception if an attempt to instantiate the DatasetLoader class is made.

        Raises:
            Exception: An error indicating that this class should not be instantiated.
        """
        raise Exception("This class should not be instantiated.")

    @staticmethod
    def get_dataset_split_names(dataset_name: str) -> List[str]:
        """
        Retrieve the names of the splits in the specified dataset.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            List[str]: A list of split names in the specified dataset.
        """
        return get_dataset_split_names(dataset_name)

    @staticmethod
    def get_tweet_topic_single_dataset() -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and split the Tweet Topic Single dataset into training, validation, and testing sets.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and testing datasets,
             respectively.
        """
        train_data: Dataset = load_dataset(DatasetLoader.TWEET_TOPIC_SINGLE,
                                           split=DatasetLoader.TWEET_TOPIC_SINGLE_TRAIN_SPLIT)
        test_data: Dataset = load_dataset(DatasetLoader.TWEET_TOPIC_SINGLE,
                                          split=DatasetLoader.TWEET_TOPIC_SINGLE_TEST_SPLIT)
        train_valid_split: DatasetDict = train_data.train_test_split(test_size=0.2)
        train_data: Dataset = train_valid_split["train"]
        valid_data: Dataset = train_valid_split["test"]
        return train_data, valid_data, test_data
