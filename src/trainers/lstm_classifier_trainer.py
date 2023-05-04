from datasets import Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.lstm_model import LSTMModel
from utils.dataset_loader import DatasetLoader
from utils.data_processing_utils import DataProcessingUtils


class LSTMClassifierTrainer:
    __optimizer: Adam
    __loss_fn: CrossEntropyLoss

    def __init__(self, model: LSTMModel, train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                 device: str = "cpu"):
        self.__model = model
        self.__optimizer = Adam(params=self.__model.parameters(), lr=0.01)
        self.__loss_fn = CrossEntropyLoss().to(device)

    def train(self, model: LSTMModel, data):
        pass

    def __collate(self, batch):
        raise NotImplementedError

    def run(self):
        train_data, valid_data, test_data = DatasetLoader.get_dataset("tweet_topic_single")

        # Standardise the data
        max_tokens = 600
        train_data = train_data.map(
            lambda x: {"text": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        valid_data = valid_data.map(
            lambda x: {"text": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        test_data = test_data.map(
            lambda x: {"text": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})

        # Create the vocabulary

        # Convert the data to tensors
        train_data = train_data.with_format(type="torch", columns=["text", "label"])
        valid_data = valid_data.with_format(type="torch", columns=["text", "label"])
        test_data = test_data.with_format(type="torch", columns=["text", "label"])

        # Create the dataloaders
        train_dataloader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_data, batch_size=128)
        test_dataloader = DataLoader(dataset=test_data, batch_size=128)
