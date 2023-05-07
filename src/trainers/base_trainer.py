from abc import ABC, abstractmethod
from typing import Tuple

from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.base_model import BaseModel


class BaseTrainer(ABC):
    _device: str
    _train_dataloader: DataLoader
    _valid_dataloader: DataLoader
    _test_dataloader: DataLoader

    def __init__(self, device: str = "cpu", train_dataloader: DataLoader = None, valid_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None) -> None:
        super().__init__()
        self._device = device
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader

    @abstractmethod
    def train(self, model: BaseModel, dataloader: DataLoader, optimizer: Optimizer, loss_fn: CrossEntropyLoss) \
            -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: BaseModel, dataloader: DataLoader, loss_fn: CrossEntropyLoss) \
            -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, model: BaseModel, epochs: int = 5, batch_size: int = 128, learning_rate: float = 0.01,
            max_tokens: int = 600) -> None:
        """
        Run the entire pipeline, including training, validation, testing, and saving.
        :param model: The model to train.
        :param epochs: Number of epochs to train for.
        :param batch_size: Batch size.
        :param learning_rate: Learning rate.
        :param max_tokens: Maximum number of tokens per tweet.
        :return: None.
        """
        raise NotImplementedError
