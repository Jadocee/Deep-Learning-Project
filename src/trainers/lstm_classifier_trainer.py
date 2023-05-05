from typing import List, Tuple

from nltk.lm import Vocabulary
from numpy import mean
from torch import Tensor, no_grad
from torch import sum as t_sum
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.lstm_model import LSTMModel
from trainers.base_trainer import BaseTrainer
from utils.data_processing_utils import DataProcessingUtils
from utils.dataset_loader import DatasetLoader
from utils.definitions import TWEET_TOPIC_SINGLE, TWEET_TOPIC_SINGLE_TRAIN_SPLIT, TWEET_TOPIC_SINGLE_TEST_SPLIT


class LSTMClassifierTrainer(BaseTrainer):

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)

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

    def run(self, model: LSTMModel, epochs: int = 5, batch_size: int = 128, learning_rate: float = 0.01,
            max_tokens: int = 600) -> None:
        train_data, valid_data, test_data = DatasetLoader.get_dataset(
            dataset_name=TWEET_TOPIC_SINGLE,
            train_split=TWEET_TOPIC_SINGLE_TRAIN_SPLIT,
            test_split=TWEET_TOPIC_SINGLE_TEST_SPLIT
        )

        # Standardise the data
        train_data = train_data.map(
            lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        valid_data = valid_data.map(
            lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        test_data = test_data.map(
            lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})

        # Create the vocabulary
        vocab: Vocabulary = DataProcessingUtils.create_vocab_1(word_tokens=train_data["tokens"])

        # Vectorise the data using vocabulary indexing
        train_data = train_data.map(
            lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})
        valid_data = valid_data.map(
            lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})
        test_data = test_data.map(
            lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})

        # Convert the data to tensors
        train_data = train_data.with_format(type="torch", columns=["ids", "label"])
        valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
        test_data = test_data.with_format(type="torch", columns=["ids", "label"])

        # Create the dataloaders
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch_size)
        test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)

        # Train the model and evaluate on the validation set
        loss_fn: CrossEntropyLoss = CrossEntropyLoss()
        optimizer: Adam = Adam(params=model.parameters(), lr=learning_rate)
        train_losses: List[float] = list()
        train_accuracies: List[float] = list()
        valid_losses: List[float] = list()
        valid_accuracies: List[float] = list()
        for epoch in range(epochs):
            train_loss, train_acc = self.train(model=model, dataloader=train_dataloader, optimizer=optimizer,
                                               loss_fn=loss_fn)
            valid_loss, valid_acc = self.evaluate(model=model, dataloader=valid_dataloader, loss_fn=loss_fn)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)
            print(f"Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc * 100:.2f}%, "
                  f"Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_acc * 100:.2f}%")

        # Evaluate the model on the test set
        test_loss, test_acc = self.evaluate(model=model, dataloader=test_dataloader, loss_fn=loss_fn)
        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc * 100:.2f}%")

        # TODO: Save performance metrics to dictionary
