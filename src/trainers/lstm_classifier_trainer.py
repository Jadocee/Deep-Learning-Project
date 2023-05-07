from typing import List, Tuple

from numpy import mean
from torch import Tensor, no_grad
from torch import sum as t_sum
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.lstm_model import LSTMModel
from trainers.base_trainer import BaseTrainer


class LSTMClassifierTrainer(BaseTrainer):

    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader,
                 device: str = "cpu") -> None:
        super().__init__(device=device, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader)

    def train(self, model: LSTMModel, dataloader: DataLoader, optimizer: Adam, loss_fn: CrossEntropyLoss) \
            -> Tuple[float, float]:
        model.train()
        losses: List[float] = list()
        accuracies: List[float] = list()
        for batch in dataloader:
            x: Tensor = batch["ids"].to(self._device)
            y: Tensor = batch["label"].to(self._device)
            optimizer.zero_grad()
            y_pred: Tensor = model.forward(x)
            loss: Tensor = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
            accuracy: Tensor = t_sum((y_pred.argmax(dim=1) == y)) / y.shape[0]
            accuracies.append(accuracy.detach().cpu().numpy())
        return mean(losses), mean(accuracies)

    def evaluate(self, model: LSTMModel, dataloader: DataLoader, loss_fn: CrossEntropyLoss) \
            -> Tuple[float, float]:
        model.eval()
        losses: List[float] = list()
        accuracies: List[float] = list()
        with no_grad():
            for batch in dataloader:
                x: Tensor = batch["ids"].to(self._device)
                y: Tensor = batch["label"].to(self._device)
                y_pred: Tensor = model.forward(x)
                loss: Tensor = loss_fn(y_pred, y)
                losses.append(loss.detach().cpu().numpy())
                accuracy: Tensor = t_sum((y_pred.argmax(dim=1) == y)) / y.shape[0]
                accuracies.append(accuracy.detach().cpu().numpy())
        return mean(losses), mean(accuracies)

    def test(self):
        raise NotImplementedError

    def __collate(self, batch) -> dict:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def run(self, model: LSTMModel, test: bool = False, epochs: int = 5, batch_size: int = 128,
            learning_rate: float = 0.01,
            max_tokens: int = 600) -> float:

        # Train the model and evaluate on the validation set
        loss_fn: CrossEntropyLoss = CrossEntropyLoss()
        optimizer: Adam = Adam(params=model.parameters(), lr=learning_rate)
        train_losses: List[float] = list()
        train_accuracies: List[float] = list()
        valid_losses: List[float] = list()
        valid_accuracies: List[float] = list()
        for epoch in range(epochs):
            train_loss, train_acc = self.train(model=model, dataloader=self._train_dataloader, optimizer=optimizer,
                                               loss_fn=loss_fn)
            valid_loss, valid_acc = self.evaluate(model=model, dataloader=self._valid_dataloader, loss_fn=loss_fn)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)
            print(f"Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc * 100:.2f}%, "
                  f"Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_acc * 100:.2f}%")

        # Evaluate the model on the test set
        if test:
            test_loss, test_acc = self.evaluate(model=model, dataloader=self._test_dataloader, loss_fn=loss_fn)
            print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc * 100:.2f}%")
            return test_acc

        return valid_accuracies[-1]

        # TODO: Save performance metrics to dict
