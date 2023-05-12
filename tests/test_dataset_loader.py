from unittest import TestCase

from datasets import load_dataset_builder, load_dataset


class TestDatasetLoader(TestCase):
    def test_get_tweet_topic_dataset(self):
        ds_builder = load_dataset_builder("cardiffnlp/tweet_topic_single")
        print(ds_builder.info.description)
        print(ds_builder.info.features)
        train_data, test_data = load_dataset("cardiffnlp/tweet_topic_single",
                                             split=["train_coling2022", "test_coling2022"])
        print(train_data)
        print(test_data)
        assert False
        # train_data, valid_data, test_data = DatasetLoader.get_dataset(
        #     dataset_name=TWEET_TOPIC_SINGLE,
        #     train_split=TWEET_TOPIC_SINGLE_TRAIN_SPLIT,
        #     test_split=TWEET_TOPIC_SINGLE_TEST_SPLIT
        # )
        # print(train_data)
        # print(valid_data)
        # print(test_data)
