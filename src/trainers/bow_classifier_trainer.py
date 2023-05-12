from typing import List, Tuple

from datasets import Dataset
from nltk.lm import Vocabulary
from numpy import mean, ndarray
from sklearn.model_selection import train_test_split
from torch import Tensor, no_grad
from torch import sum as t_sum
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.bow_model import BOWModel
import numpy as np
from models.lstm_model import LSTMModel
from utils.data_processing_utils import DataProcessingUtils
from utils.dataset_loader import DatasetLoader


class BOWClassifierTrainer:
    __device: str

    def __init__(self, device: str = "cpu") -> None:
        self.__device = device

    def evaluate(model, dataloader, loss_fn, device):
        model.eval()
        losses, accuracies = [], []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['multi_hot'].to(device)
                labels = batch['label'].to(device)
                # Forward pass
                preds = model(inputs)
                print(len(preds))
                # Calculate loss
                loss = loss_fn(preds, labels)
                # Log
                losses.append(loss.detach().cpu().numpy())
            accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
            accuracies.append(accuracy.detach().cpu().numpy())
        return np.mean(losses), np.mean(accuracies)
    
    def train(model, dataloader, loss_fn, optimizer, device):
        model.train()
        losses, accuracies = [], []
        for batch in dataloader:
            inputs = batch['multi_hot'].to(device)
            labels = batch['label'].to(device)
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


    def run(self, epochs: int = 5, batch_size: int = 128, learning_rate: float = 0.01,
            max_tokens: int = 600) -> None:
        train_data, valid_data, test_data = DatasetLoader.get_tweet_topic_single_dataset()
        # Standardise the data
        train_data = train_data.map(
            lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        valid_data = valid_data.map(
            lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        test_data = test_data.map(
            lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})

        # # Create the vocabulary
        vocab: Vocabulary = DataProcessingUtils.create_vocab_2(train_data,valid_data,test_data)

        # Vectorise the data using vocabulary indexing
        # train_data = train_data.map(
        #     lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})
        # valid_data = valid_data.map(
        #     lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})
        # test_data = test_data.map(
        #     lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})

        # Numericalize the data
        train_data = train_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})
        valid_data = valid_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})
        test_data = test_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})


        train_data = train_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
        valid_data = valid_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
        test_data = test_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})

        # Convert the data to tensors
        train_data = train_data.with_format(type="torch", columns=["multi_hot", "label"])
        valid_data = valid_data.with_format(type="torch", columns=["multi_hot", "label"])
        test_data = test_data.with_format(type="torch", columns=["multi_hot", "label"])

        # Create the dataloaders
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch_size)
        test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)

        print(len(vocab))
        model = BOWModel(vocab_size=len(vocab)).to(self.__device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = CrossEntropyLoss().to(self.__device)
        # Train the model and evaluate on the validation set
        train_losses, train_accuracies = [], []
        valid_losses, valid_accuracies = [], []
        for epoch in range(10):
            # Train
            train_loss, train_accuracy = BOWClassifierTrainer.train(model, train_dataloader, loss_fn, optimizer, self.__device)
            # Evaluate
            valid_loss, valid_accuracy = BOWClassifierTrainer.evaluate(model, valid_dataloader, loss_fn, self.__device)
            # Log
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)
            print("Epoch {}: train_loss={:.4f}, train_accuracy={:.4f}, valid_loss={:.4f}, valid_accuracy={:.4f}".format(
                epoch+1, train_loss, train_accuracy, valid_loss, valid_accuracy))