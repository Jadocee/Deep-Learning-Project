from typing import List, Tuple, Optional

from numpy import mean, float64, ndarray
from optuna import Trial
from optuna.exceptions import TrialPruned
from torch import Tensor, no_grad
from torch import sum as t_sum
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from models.lstm_model import LSTMModel
from trainers.base_trainer import BaseTrainer


class LSTMClassifierTrainer(BaseTrainer):
    __vocab: Vocab
    __pad_index: int

    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader,
                 vocab: Vocab, device: str = "cpu") -> None:
        super().__init__(device=device, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader)
        self.__vocab = vocab
        self.__pad_index = vocab["<pad>"]

    def train(self, model: LSTMModel, dataloader: DataLoader, optimiser: Adam, loss_fn: CrossEntropyLoss) \
            -> Tuple[float64, float64]:
        model.train()
        losses: List[ndarray] = list()
        accuracies: List[ndarray] = list()
        for batch in dataloader:
            x: Tensor = batch["ids"].to(self._device)
            y: Tensor = batch["label"].to(self._device)
            optimiser.zero_grad()
            y_pred: Tensor = model.forward(x)
            loss: Tensor = loss_fn(y_pred, y)
            loss.backward()
            optimiser.step()
            losses.append(loss.detach().cpu().numpy())
            accuracy: Tensor = t_sum((y_pred.argmax(dim=1) == y)) / y.shape[0]
            accuracies.append(accuracy.detach().cpu().numpy())
        return mean(losses), mean(accuracies)

    def evaluate(self, model: LSTMModel, dataloader: DataLoader, loss_fn: CrossEntropyLoss) \
            -> Tuple[float64, float64]:
        model.eval()
        losses: List[ndarray] = list()
        accuracies: List[ndarray] = list()
        with no_grad():
            for batch in dataloader:
                x: Tensor = batch["ids"].to(self._device)
                y: Tensor = batch["label"].to(self._device)
                y_pred: Tensor = model.forward(x)
                loss: Tensor = loss_fn(y_pred, y)
                losses.append(loss.detach().cpu().numpy())
                accuracy: Tensor = t_sum((y_pred.argmax(dim=1) == y)) / y.shape[0]
                accuracies.append(accuracy.detach().cpu().numpy())
        print(f"Loss: {mean(losses)} Accuracy: {mean(accuracies)}")
        return mean(losses), mean(accuracies)

    def test(self, model: LSTMModel):
        loss_fn: CrossEntropyLoss = CrossEntropyLoss()
        test_loss, test_acc = self.evaluate(model=model, dataloader=self._test_dataloader, loss_fn=loss_fn)
        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc * 100:.2f}%")

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def run(self,
            model: LSTMModel,
            epochs: int = 5,
            batch_size: int = 128,
            learning_rate: float = 0.01,
            max_tokens: int = 600,
            trial: Optional[Trial] = None,
            ) -> float:

        # Verify parameters
        if (epochs < 1) or (not isinstance(epochs, int)):
            raise ValueError(f"Invalid epochs: {epochs}")
        if (batch_size < 1) or (not isinstance(batch_size, int)):
            raise ValueError(f"Invalid batch size: {batch_size}")
        if (learning_rate < 0.0) or (not isinstance(learning_rate, float)):
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if (max_tokens < 1) or (not isinstance(max_tokens, int)):
            raise ValueError(f"Invalid max tokens: {max_tokens}")

        # Train the model on the training set and evaluate on the validation set
        loss_fn: CrossEntropyLoss = CrossEntropyLoss()
        optimiser: Adam = Adam(params=model.parameters, lr=learning_rate)
        train_losses: List[float] = list()
        train_accuracies: List[float] = list()
        valid_losses: List[float] = list()
        valid_accuracies: List[float] = list()

        for epoch in range(epochs):
            train_loss, train_acc = self.train(model=model, dataloader=self._train_dataloader, optimiser=optimiser,
                                               loss_fn=loss_fn)
            valid_loss, valid_acc = self.evaluate(model=model, dataloader=self._valid_dataloader, loss_fn=loss_fn)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)

            if trial:
                trial.report(valid_accuracies[-1], epoch)
                if trial.should_prune():
                    raise TrialPruned()
            else:
                print(f"Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc * 100:.2f}%, "
                      f"Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_acc * 100:.2f}%")

        return valid_accuracies[-1]

        # TODO: Save performance metrics to dict
