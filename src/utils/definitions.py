from os.path import dirname, abspath

ROOT: str = dirname(dirname(dirname(abspath(__file__))))
"""
The root directory of the project.
"""

DATA_DIR: str = f"{ROOT}/data"
"""
The directory where the datasets are stored.
"""

OUT_DIR: str = f"{ROOT}/out"
"""
The directory where the output files are stored.
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