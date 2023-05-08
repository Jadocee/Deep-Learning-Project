from typing import List, Tuple

from datasets import load_dataset, get_dataset_split_names, Dataset, DatasetDict


class DatasetLoader:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_dataset_split_names(dataset_name: str) -> List[str]:
        return get_dataset_split_names(dataset_name)

    @staticmethod
    def get_dataset(dataset_name: str, train_split: str, test_split: str, valid_size: float = 0.2) \
            -> Tuple[Dataset, Dataset, Dataset]:
        train_data: Dataset = load_dataset(dataset_name, split=train_split)
        test_data: Dataset = load_dataset(dataset_name, split=test_split)
        train_valid_split: DatasetDict = train_data.train_test_split(test_size=valid_size)
        train_data: Dataset = train_valid_split["train"]
        valid_data: Dataset = train_valid_split["test"]
        return train_data, valid_data, test_data
