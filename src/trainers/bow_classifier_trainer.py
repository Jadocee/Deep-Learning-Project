from typing import Dict, Optional

import numpy as np
import torch
from optuna import Trial
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from models.bow_model import BOWModel
from trainers.base_trainer import BaseTrainer


class BOWClassifierTrainer(BaseTrainer):

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader,
                 vocab: Vocab, device: str = "cpu") -> None:
        super().__init__(device=device, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader)
        self.__vocab = vocab
        self.__pad_index = vocab["<pad>"]

    def evaluate(self, model: BOWModel, dataloader, loss_fn):
        model.eval()
        losses, accuracies = [], []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['multi_hot']
                labels = batch['label']
                # Forward pass
                preds = model(inputs)
                # Calculate loss
                loss = loss_fn(preds, labels)
                # Log
                losses.append(loss.detach().cpu().numpy())
            accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
            accuracies.append(accuracy.detach().cpu().numpy())
        return np.mean(losses), np.mean(accuracies)

    def train(self, model: BOWModel, dataloader, loss_fn, optimizer):
        model.train()
        losses, accuracies = [], []
        for batch in dataloader:
            inputs = batch['multi_hot']
            labels = batch['label']
            # Reset the gradients for all variables
            optimizer.zero_grad()
            # Forward pass
            preds = model(inputs)
            # Calculate loss
            loss = loss_fn(preds, labels)
            # Backward pass
            loss.backward()
            # Adjust weights
            optimizer.step()
            # Log
            losses.append(loss.detach().cpu().numpy())
            accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
            accuracies.append(accuracy.detach().cpu().numpy())
        return np.mean(losses), np.mean(accuracies)

    def run(self,
            model: BOWModel,
            epochs: int = 5,
            batch_size: int = 128,
            learning_rate: float = 0.01,
            max_tokens: int = 600,
            trial: Optional[Trial] = None,
            optimiser_name: str = "Adam",
            lr_scheduler_name: Optional[str] = None,
            kwargs: Optional[Dict] = None
            ) -> float:

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = CrossEntropyLoss()
        # Train the model and evaluate on the validation set
        train_losses, train_accuracies = [], []
        valid_losses, valid_accuracies = [], []
        for epoch in range(10):
            # Train
            train_loss, train_accuracy = self.train(model, self._train_dataloader, loss_fn, optimizer)
            # Evaluate
            valid_loss, valid_accuracy = self.evaluate(model, self._valid_dataloader, loss_fn)
            # Log
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)
            print("Epoch {}: train_loss={:.4f}, train_accuracy={:.4f}, valid_loss={:.4f}, valid_accuracy={:.4f}".format(
                epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy))

        return valid_accuracies[-1]
