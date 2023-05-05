from abc import ABC, abstractmethod
from typing import Tuple

from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.base_model import BaseModel


class BaseTrainer(ABC):

    _device: str

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.__device = device

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
        raise NotImplementedError
