from typing import List, Tuple

from datasets import load_dataset, get_dataset_split_names, Dataset, DatasetDict


class DatasetLoader:
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
        pass

    @staticmethod
    def get_dataset_split_names(dataset_name: str) -> List[str]:
        return get_dataset_split_names(dataset_name)

    @staticmethod
    def get_tweet_topic_single_dataset() -> Tuple[Dataset, Dataset, Dataset]:
        train_data: Dataset = load_dataset(DatasetLoader.TWEET_TOPIC_SINGLE,
                                           split=DatasetLoader.TWEET_TOPIC_SINGLE_TRAIN_SPLIT)
        test_data: Dataset = load_dataset(DatasetLoader.TWEET_TOPIC_SINGLE,
                                          split=DatasetLoader.TWEET_TOPIC_SINGLE_TEST_SPLIT)
        train_valid_split: DatasetDict = train_data.train_test_split(test_size=0.2)
        train_data: Dataset = train_valid_split["train"]
        valid_data: Dataset = train_valid_split["test"]
        return train_data, valid_data, test_data
