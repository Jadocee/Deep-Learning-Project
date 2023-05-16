from typing import List, Tuple, Optional, Union, Dict, Any

import torch.optim as optim
from numpy import mean, float64, ndarray
from optuna import Trial
from optuna.exceptions import TrialPruned
from pandas import DataFrame
from torch import Tensor, no_grad
from torch import sum as t_sum
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
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
        __pad_index (int): The index used for padding in the dataset.
    """
    __vocab: Vocab
    __pad_index: int

    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader,
                 vocab: Vocab, device: str = "cpu") -> None:
        """
        Initializes LSTMClassifierTrainer with training, validation, and testing data, a vocabulary, and a device.

        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
            valid_dataloader (DataLoader): DataLoader for the validation data.
            test_dataloader (DataLoader): DataLoader for the testing data.
            vocab (Vocab): A torchtext Vocab object for the vocabulary of the dataset.
            device (str, optional): Device to train the model on. Default is "cpu".
        """
        super().__init__(device=device, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader)
        self.__vocab = vocab
        self.__pad_index = vocab["<pad>"]

    def train(self, model: LSTMModel, dataloader: DataLoader, loss_fn: CrossEntropyLoss, optimiser: Optimizer) \
            -> Tuple[float64, float64]:
        """
        Trains an LSTM model using a dataloader, a loss function, and an optimizer.

        Args:
            model (LSTMModel): The model to train.
            dataloader (DataLoader): DataLoader for providing the training data.
            loss_fn (CrossEntropyLoss): The loss function to use during training.
            optimiser (Optimizer): The optimizer to use during training.

        Returns:
            Tuple[float64, float64]: The mean loss and mean accuracy over the training data.
        """
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
        """
        Evaluates an LSTM model using a dataloader and a loss function.

        Args:
            model (LSTMModel): The model to evaluate.
            dataloader (DataLoader): DataLoader for providing the evaluation data.
            loss_fn (CrossEntropyLoss): The loss function to use during evaluation.

        Returns:
            Tuple[float64, float64]: The mean loss and mean accuracy over the evaluation data.
        """
        model.eval()
        losses: List[ndarray] = list()
        accuracies: List[ndarray] = list()
        # predicted_values: ndarray = ndarray(shape=(0,), dtype=float)
        # actual_values: ndarray = ndarray(shape=(0,), dtype=float)
        with no_grad():
            for batch in dataloader:
                x: Tensor = batch["ids"].to(self._device)
                y: Tensor = batch["label"].to(self._device)
                output: Tensor = model.forward(x)
                loss: Tensor = loss_fn(output, y)
                losses.append(loss.detach().cpu().numpy())
                # accuracy: Tensor = t_sum((y_pred.argmax(dim=1) == y)) / y.shape[0]
                y_pred: Tensor = output.argmax(dim=1)
                accuracy: Tensor = t_sum((y_pred == y)) / y.shape[0]
                accuracies.append(accuracy.detach().cpu().numpy())
                # predicted_values = concatenate((predicted_values, y_pred.detach().cpu().numpy()))
                # actual_values = concatenate((actual_values, y.detach().cpu().numpy()))
        return mean(losses), mean(accuracies)

    def test(self, model: LSTMModel):
        """
        Tests an LSTM model using the test data.

        Args:
            model (LSTMModel): The model to test.
        """
        loss_fn: CrossEntropyLoss = CrossEntropyLoss()
        test_loss, test_acc = self.evaluate(model=model, dataloader=self._test_dataloader, loss_fn=loss_fn)
        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc * 100:.2f}%")

    def run(self,
            model: LSTMModel,
            epochs: int = 5,
            batch_size: int = 128,
            learning_rate: float = 1e-2,
            max_tokens: int = 600,
            trial: Optional[Trial] = None,
            optimiser_name: str = "Adam",
            lr_scheduler_name: Optional[str] = None,
            kwargs: Optional[Dict[str, Any]] = None
            ) -> float:
        """
        Runs the entire process of training and validating the model.

        Args:
            model (LSTMModel): The model to train and validate.
            epochs (int, optional): Number of epochs to run for training. Default is 5.
            batch_size (int, optional): Size of the batches for the DataLoader. Default is 128.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-2.
            max_tokens (int, optional): Maximum number of tokens for the LSTM model. Default is 600.
            trial (Trial, optional): Optional Optuna trial object for hyperparameter optimization. Default is None.
            optimiser_name (str, optional): Name of the optimizer to be used for training. Default is "Adam".
            lr_scheduler_name (str, optional): Name of the learning rate scheduler to be used for training. Default is
            None.
            kwargs (dict, optional): Additional keyword arguments for the learning rate scheduler. Default is None.

        Returns:
            float: The accuracy of the model on the validation set at the final epoch.
        """
        # TODO: Move this method to the base trainer class

        # Verify parameters
        if (epochs < 1) or (not isinstance(epochs, int)):
            raise ValueError(f"Invalid epochs: {epochs}")
        if (batch_size < 1) or (not isinstance(batch_size, int)):
            raise ValueError(f"Invalid batch size: {batch_size}")
        if (learning_rate < 0.0) or (not isinstance(learning_rate, float)):
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if (max_tokens < 1) or (not isinstance(max_tokens, int)):
            raise ValueError(f"Invalid max tokens: {max_tokens}")

        # Set up the optimiser and scheduler
        optimiser: Optimizer = getattr(optim, optimiser_name)(model.parameters, lr=learning_rate)
        scheduler: Union[LRScheduler, None] = getattr(lr_scheduler, lr_scheduler_name)(optimiser, **kwargs) \
            if lr_scheduler_name else None

        loss_fn: CrossEntropyLoss = CrossEntropyLoss()
        train_losses: List[float] = list()
        train_accuracies: List[float] = list()
        valid_losses: List[float] = list()
        valid_accuracies: List[float] = list()

        # Train the model on the training set and evaluate on the validation set
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

            df: DataFrame = DataFrame({"Epoch": f"{epoch + 1:02}/{epochs:02}", "Train Loss": f"{train_loss:.3f}",
                                       "Train Accuracy": f"{train_acc * 100:.2f}%", "Valid Loss": f"{valid_loss:.3f}",
                                       "Valid Accuracy": f"{valid_acc * 100:.2f}%"}, index=[0])
            print(df.to_string(index=False, header=(epoch == 0), justify="center", col_space=15))

            # print(f"Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc * 100:.2f}%, "
            #       f"Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_acc * 100:.2f}%")

            if scheduler:
                scheduler.step(valid_loss) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

        return valid_accuracies[-1]

        # TODO: Save performance metrics to dict
