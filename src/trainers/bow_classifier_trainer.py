from typing import Dict, List, Optional

import numpy as np
from pandas import DataFrame
import torch
from torch import Tensor
from optuna import Trial
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from torch.optim import Optimizer
from torch import sum as t_sum

from models.bow_model import BOWModel
from trainers.base_trainer import BaseTrainer
from utils.hyperparam_utils import HyperParamUtils


class BOWClassifierTrainer(BaseTrainer):
    __vocab: Vocab
    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader,
                 vocab: Vocab, device: str = "cpu") -> None:
        super().__init__(device=device, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader, loss_fn=CrossEntropyLoss())
        self.__vocab = vocab
        self.__pad_index = vocab["<pad>"]

    def _evaluate(self,model:BOWModel, dataloader, loss_fn = CrossEntropyLoss()):
        model.eval()
        losses, accuracies = [], []
        predictions: np.ndarray = np.ndarray(shape=(0,), dtype=int)
        targets: np.ndarray = np.ndarray(shape=(0,), dtype=int)
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['ids'].to(self._device)
                labels = batch['label'].to(self._device)
                # Forward pass
                preds = model.forward(inputs)
                # Calculate loss
                loss = loss_fn(preds, labels)
                # Log
                losses.append(loss.detach().cpu().numpy())
                y_pred: Tensor = preds.argmax(dim=1)
                accuracy: Tensor = t_sum((y_pred == labels)) / labels.shape[0]
                accuracies.append(accuracy.detach().cpu().numpy())
                predictions = np.concatenate((predictions, y_pred.detach().cpu().numpy()))
                targets = np.concatenate((targets, labels.detach().cpu().numpy()))

        return np.mean(losses), np.mean(accuracies), predictions, targets

    def _train(self, model: BOWModel, optimiser: Optimizer):
        model.train()
        loss_fn = CrossEntropyLoss()
        losses: List[np.ndarray] = list()
        accuracies: List[np.ndarray] = list()
        for batch in self._train_dataloader:
            inputs = batch['ids'].to(self._device)
            labels = batch['label'].to(self._device)
            # Reset the gradients for all variables
            optimiser.zero_grad()
            # Forward pass
            preds = model.forward(inputs)
            # Calculate loss
            loss = loss_fn(preds, labels)
            # Backward pass
            loss.backward()
            # Adjust weights
            optimiser.step()
            # Log
            losses.append(loss.detach().cpu().numpy())
            y_pred: Tensor = preds.argmax(dim=1)
            accuracy: Tensor = t_sum((y_pred == labels)) / labels.shape[0]
            accuracies.append(accuracy.detach().cpu().numpy())
            
        return np.mean(losses), np.mean(accuracies)

    def run(self,
            model: BOWModel,
            epochs: int = 5,
            learning_rate: float = 0.01,
            optimiser_name: str = "Adam",
            ) -> float:

        optimiser: Optimizer = HyperParamUtils.define_optimiser(
            optimiser_name=optimiser_name,
            model_params=model.get_parameters(),
            learning_rate=learning_rate
            )
        loss_fn = CrossEntropyLoss()
        # Train the model and evaluate on the validation set
        train_losses: List[float] = list()
        train_accuracies: List[float] = list()
        valid_losses: List[float] = list()
        valid_accuracies: List[float] = list()
        for epoch in range(epochs):
            # Train
            train_loss, train_accuracy = self._train(model, self._train_dataloader, loss_fn, optimiser)
            # Evaluate
            valid_loss, valid_accuracy = self._evaluate(model, self._valid_dataloader, loss_fn)
            # Log
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            df: DataFrame = DataFrame({"Epoch": f"{epoch + 1:02}/{epochs:02}", "Train Loss": f"{train_loss:.3f}",
                                       "Train Accuracy": f"{train_accuracy * 100:.2f}%", "Valid Loss": f"{valid_loss:.3f}",
                                       "Valid Accuracy": f"{valid_accuracy * 100:.2f}%"}, index=[0])

            print(df.to_string(index=False, header=(epoch == 0), justify="center", col_space=15))

            # print("Epoch {}: train_loss={:.4f}, train_accuracy={:.4f}, valid_loss={:.4f}, valid_accuracy={:.4f}".format(
            #     epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy))

        return valid_accuracies[-1],valid_losses[-1]

