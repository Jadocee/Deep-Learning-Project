from datasets import load_dataset


class DatasetLoader:

    def __init__(self):
        pass

    @staticmethod
    def get_dataset(dataset_name: str):
        if dataset_name == "tweet_topic_single":
            train_data = load_dataset(dataset_name, split="train_coling2022")
            test_data = load_dataset(dataset_name, split="test_coling2022")
            train_valid_split = train_data.train_test_split(test_size=0.2)
            train_data = train_valid_split["train"]
            valid_data = train_valid_split["test"]
            # train_data = train_data.with_format(type="torch", columns=["text", "date", "label", "label_name", "id"])
            # valid_data = valid_data.with_format(type="torch", columns=["text", "date", "label", "label_name", "id"])
            # test_data = test_data.with_format(type="torch", columns=["text", "date", "label", "label_name", "id"])

            return train_data, valid_data, test_data
        else:
            raise Exception("Unknown dataset")
