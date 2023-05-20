from abc import ABC, abstractmethod
from typing import Tuple

from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.base_model import BaseModel


class BaseTrainer(ABC):
    """
    Abstract Base Class for model training.

    Attributes:
        _device (str): Device to be used for computations.
        _train_dataloader (DataLoader): DataLoader for the training set.
        _valid_dataloader (DataLoader): DataLoader for the validation set.
        _test_dataloader (DataLoader): DataLoader for the test set.
    """
    _device: str
    _train_dataloader: DataLoader
    _valid_dataloader: DataLoader
    _test_dataloader: DataLoader

    def __init__(self, device: str = "cpu", train_dataloader: DataLoader = None, valid_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None) -> None:
        """
        Initializes the BaseTrainer class.

        Args:
            device (str, optional): The device to use for computations. Defaults to "cpu".
            train_dataloader (DataLoader, optional): DataLoader for the training set. Defaults to None.
            valid_dataloader (DataLoader, optional): DataLoader for the validation set. Defaults to None.
            test_dataloader (DataLoader, optional): DataLoader for the test set. Defaults to None.
        """
        super().__init__()
        self._device = device
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader

    @abstractmethod
    def train(self, model: BaseModel, dataloader: DataLoader, optimizer: Optimizer, loss_fn: CrossEntropyLoss) \
            -> Tuple[float, float]:
        """
        Train the model.

        Args:
            model (BaseModel): The model to train.
            dataloader (DataLoader): DataLoader for the dataset to train on.
            optimizer (Optimizer): The optimizer to use during training.
            loss_fn (CrossEntropyLoss): The loss function to use during training.

        Returns:
            Tuple[float, float]: The loss and accuracy on the training set.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: BaseModel, dataloader: DataLoader, loss_fn: CrossEntropyLoss) \
            -> Tuple[float, float]:
        """
        Evaluate the model.

        Args:
            model (BaseModel): The model to evaluate.
            dataloader (DataLoader): DataLoader for the dataset to evaluate on.
            loss_fn (CrossEntropyLoss): The loss function to use during evaluation.

        Returns:
            Tuple[float, float]: The loss and accuracy on the evaluation set.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, model: BaseModel, epochs: int = 5, batch_size: int = 128, learning_rate: float = 0.01,
            max_tokens: int = 600) -> None:
        """
        Run the entire pipeline, including training, validation, testing, and saving.

        Args:
            model (BaseModel): The model to train.
            epochs (int, optional): Number of epochs to train for. Defaults to 5.
            batch_size (int, optional): Batch size. Defaults to 128.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            max_tokens (int, optional): Maximum number of tokens per tweet. Defaults to 600.

        Returns:
            None
        """
        raise NotImplementedError
