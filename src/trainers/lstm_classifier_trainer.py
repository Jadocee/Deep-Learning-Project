from typing import List, Tuple

from numpy import mean, float64, ndarray, concatenate
from torch import Tensor, no_grad
from torch import sum as t_sum
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from models.lstm_model import LSTMModel
from trainers.base_trainer import BaseTrainer


class LSTMClassifierTrainer(BaseTrainer):
    """
    The LSTMClassifierTrainer class is used for training LSTM models. It inherits from the BaseTrainer class
    and overrides its methods to provide functionality for training, evaluation, testing, and running LSTM models.

    Attributes:
        __vocab (Vocab): A torchtext Vocab object which encodes the vocabulary used in the dataset.
    """
    __vocab: Vocab

    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 test_dataloader: DataLoader, vocab: Vocab, device: str = "cpu"):
        """
        Initializes LSTMClassifierTrainer with training, validation, and testing data, a vocabulary, and a device.

        Args:
            device (str, optional): Device to train the model on. Default is "cpu".
        """
        super().__init__(device=device, loss_fn=CrossEntropyLoss(), train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader, test_dataloader=test_dataloader)
        self.__vocab = vocab

    def _train(self, model: LSTMModel, optimiser: Optimizer) -> Tuple[float64, float64]:
        """
        Trains an LSTM model using a dataloader, a loss function, and an optimizer.

        Args:
            model (LSTMModel): The model to train.
            optimiser (Optimizer): The optimizer to use during training.

        Returns:
            Tuple[float64, float64]: The mean loss and mean accuracy over the training data.
        """
        model.train()
        losses: List[ndarray] = list()
        accuracies: List[ndarray] = list()
        for batch in self._train_dataloader:
            x: Tensor = batch["ids"].to(self._device)
            y: Tensor = batch["label"].to(self._device)
            optimiser.zero_grad()
            y_pred: Tensor = model.forward(x)
            loss: Tensor = self._criterion(y_pred, y)
            loss.backward()
            optimiser.step()
            losses.append(loss.detach().cpu().numpy())
            accuracy: Tensor = t_sum((y_pred.argmax(dim=1) == y)) / y.shape[0]
            accuracies.append(accuracy.detach().cpu().numpy())
        return mean(losses), mean(accuracies)

    def _evaluate(self, model: LSTMModel, dataloader: DataLoader) -> Tuple[float64, float64, ndarray, ndarray]:
        """
        Evaluates an LSTM model using a dataloader and a loss function.

        Args:
            model (LSTMModel): The model to evaluate.
            dataloader (DataLoader): DataLoader for providing the evaluation data.

        Returns:
            Tuple[float64, float64]: The mean loss and mean accuracy over the evaluation data.
        """
        model.eval()
        losses: List[ndarray] = list()
        accuracies: List[ndarray] = list()
        predictions: ndarray = ndarray(shape=(0,), dtype=int)
        targets: ndarray = ndarray(shape=(0,), dtype=int)
        with no_grad():
            for batch in dataloader:
                x: Tensor = batch["ids"].to(self._device)
                y: Tensor = batch["label"].to(self._device)
                output: Tensor = model.forward(x)
                loss: Tensor = self._criterion(output, y)
                losses.append(loss.detach().cpu().numpy())
                y_pred: Tensor = output.argmax(dim=1)
                accuracy: Tensor = t_sum((y_pred == y)) / y.shape[0]
                accuracies.append(accuracy.detach().cpu().numpy())
                predictions = concatenate((predictions, y_pred.detach().cpu().numpy()))
                targets = concatenate((targets, y.detach().cpu().numpy()))

        return mean(losses), mean(accuracies), predictions, targets
