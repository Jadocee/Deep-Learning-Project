from unittest import TestCase

from utils.dataset_loader import DatasetLoader
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names



class TestDatasetLoader(TestCase):
    def test_get_tweet_topic_dataset(self):
        ds_builder = load_dataset_builder("cardiffnlp/tweet_topic_single")
        print(ds_builder.info.description)
        print(ds_builder.info.features)




