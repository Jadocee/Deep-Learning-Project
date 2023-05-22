from abc import ABC, abstractmethod
from time import time
from typing import Tuple, Dict, List, Optional, Any

from optuna import Trial, TrialPruned
from pandas import DataFrame
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from utils.hyperparam_utils import HyperParamUtils


class BaseTrainer(ABC):
    """
    Abstract Base Class for model training.

    Attributes:
        _device (str, optional): Device to be used for computations. Modules will be moved to this device.
        _train_dataloader (DataLoader): DataLoader for the training dataset.
        _valid_dataloader (DataLoader): DataLoader for the validation dataset.
        _test_dataloader (DataLoader): DataLoader for the test dataset.
        _criterion (Module): The loss function to use during training.
    """

    _device: str
    _train_dataloader: DataLoader
    _valid_dataloader: DataLoader
    _test_dataloader: DataLoader
    _criterion: Module

    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader,
                 loss_fn: Module, device: str = "cpu"):
        """
        Initializes the BaseTrainer class.

        Args:
            device (str, optional): The device to use for computations. Defaults to "cpu".
        """
        super().__init__()
        self._device = device
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader
        self._criterion = loss_fn.to(self._device)

    @abstractmethod
    def _train(self, model: BaseModel, optimiser: Optimizer) -> Tuple[float, float]:
        """
        Train the model.

        Args:
            model (BaseModel): The model to train.
            dataloader (DataLoader): DataLoader for the dataset to train on.
            optimiser (Optimizer): The optimiser to use during training.

        Returns:
            Tuple[float, float]: The loss and accuracy on the training set.
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, model: BaseModel, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model.

        Args:
            model (BaseModel): The model to evaluate.
            dataloader (DataLoader): DataLoader for the dataset to evaluate on.

        Returns:
            Tuple[float, float]: The loss and accuracy on the evaluation set.
        """
        raise NotImplementedError

    def test(self, model: BaseModel) -> Tuple[float, float]:
        """
        Test the trained model.

        Evaluates the given model on the test set and creates a report of the results. The report includes the
        loss and accuracy on the test set, as well as the confusion matrix and classification report.

        Args:
            model (BaseModel): The trained model to test.

        Returns:
            Tuple[float, float]: The loss and accuracy on the test set.

        Raises:
            ValueError: If the given model is not trained.
        """
        if not model.is_trained():
            raise ValueError(f"Model {model} is not trained. Cannot test untrained model.")

        model.eval()
        test_loss, test_acc = self._evaluate(model, self._test_dataloader)
        # TODO: Create report
        return test_loss, test_acc

    def fit(self,
            model: BaseModel,
            epochs: int = 5,
            batch_size: int = 128,
            learning_rate: float = 1e-2,
            lr_scheduler_params: Optional[Dict[str, Any]] = None,
            trial: Optional[Trial] = None,
            optimiser_name: str = "Adam") -> Dict[str, Any]:
        """
        Fits the given model to the training data.

        Runs the training pipeline for the given model, using the given hyperparameters. The pipeline consists of
        training the model on the training set, and evaluating it on the validation set. The model is trained for
        the given number of epochs, using the given batch size and learning rate. The given optimiser is used to
        optimise the model parameters. If a learning rate scheduler is given, it is used to adjust the learning rate
        during training. If a trial is given, it is used to potentially prune the training process.

        Args:
            lr_scheduler_params (Dict[str, Any], optional): A dictionary containing the name and keyword arguments
                for the learning rate scheduler to use. Defaults to None.
            trial (Optional[Trial], optional): The optuna trial to use during training. Required if
                the relevant Optuna study uses a pruner. Defaults to None.
            optimiser_name (str, optional): The name of the optimiser to use. Defaults to "Adam".
            model (BaseModel): The model to train.
            epochs (int, optional): Number of epochs to train for. Defaults to 5.
            batch_size (int, optional): Batch size. Defaults to 128.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.

        Returns:
            None
        """
        if (epochs < 1) or (not isinstance(epochs, int)):
            raise ValueError(f"Invalid epochs: {epochs}")
        if (learning_rate < 0.0) or (not isinstance(learning_rate, float)):
            raise ValueError(f"Invalid learning rate: {learning_rate}")

        optimiser: Optimizer = HyperParamUtils.define_optimiser(
            optimiser_name=optimiser_name,
            model_params=model.get_parameters(),
            learning_rate=learning_rate
        )

        scheduler: LRScheduler = HyperParamUtils.define_scheduler(
            learning_rate_scheduler_name=lr_scheduler_params["name"],
            optimiser=optimiser,
            **lr_scheduler_params["kwargs"]
        ) if lr_scheduler_params else None

        train_losses: List[float] = list()
        train_accuracies: List[float] = list()
        valid_losses: List[float] = list()
        valid_accuracies: List[float] = list()

        print(f"Training model {model} for {epochs} epochs...")
        accumulated_time: float = 0.0
        for epoch in range(epochs):
            t_start: float = time()
            train_loss, train_acc = self._train(model=model, optimiser=optimiser)
            valid_loss, valid_acc = self._evaluate(model=model, dataloader=self._valid_dataloader)
            accumulated_time += time() - t_start
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)

            if trial:
                trial.report(valid_accuracies[-1], epoch)
                if trial.should_prune():
                    raise TrialPruned()

            message: str = ""

            if len(valid_losses) > 2:
                if valid_losses[-1] > valid_losses[-2] and valid_losses[-1] > valid_losses[-3]:
                    message = "Validation loss increasing; model might be overfitting."
                if train_losses[-1] > train_losses[-2] and train_losses[-1] > train_losses[-3]:
                    message = "Training loss is not decreasing; model might be underfitting."

            df: DataFrame = DataFrame({"Epoch": f"{epoch + 1:02}/{epochs:02}", "Train Loss": f"{train_loss:.3f}",
                                       "Train Accuracy": f"{train_acc * 100:.2f}%", "Valid Loss": f"{valid_loss:.3f}",
                                       "Valid Accuracy": f"{valid_acc * 100:.2f}%", "Message": f"{message}"}, index=[0])
            print(df.to_string(index=False, header=(epoch == 0), justify="center", col_space=15))

            if scheduler:
                scheduler.step(valid_loss) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

        model.set_trained()

        return {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "valid_losses": valid_losses,
            "valid_accuracies": valid_accuracies,
            "time": accumulated_time
        }
