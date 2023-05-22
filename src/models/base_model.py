from abc import ABC, abstractmethod
from os.path import join
from typing import Dict, Any, Iterator

import torch
from torch import Tensor
from torch.nn import ModuleList, Parameter

from utils.definitions import MODELS_DIR


class BaseModel(ABC):
    """
    Abstract base class for models. It initializes with a device and a ModuleList, and
    provides basic model functionalities including train, eval, predict, and an abstract
    forward method.

    Attributes:
        _modules (ModuleList): A ModuleList to hold the submodules.
        _device (str): The device the model is running on, defaults to "cpu".
        __trained (bool): Whether the model has been trained or not.
    """
    _modules: ModuleList
    _device: str
    __hyperparameters: Dict[str, Any]
    __trained: bool

    def __init__(self, device: str = "cpu"):
        """
        Initializes the BaseModel class.

        Args:
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super().__init__()
        self._device = device
        self._modules = ModuleList()
        self.__hyperparameters = dict()
        self.__trained = False

    def is_trained(self) -> bool:
        """
        Returns whether the model has been trained or not.

        Returns:
            bool: True if the model has been trained, False otherwise.
        """
        return self.__trained

    def set_trained(self, trained: bool = True) -> None:
        """
        Sets the trained attribute of the model.

        This method is used to set the trained attribute of the model to True after training, and should only be called
        by the trainer after the final epoch of the fitting/training process.

        Args:
            trained (bool, optional): The value to set the trained attribute to. Defaults to True.

        Returns:
            None
        """
        self.__trained = trained

    def get_parameters(self) -> Iterator[Parameter]:
        """
        Returns the parameters of the model.

        Returns:
            Iterator[Parameter]: An iterator over the parameters of the model.
        """
        return self._modules.parameters()

    def save(self, model_name: str, trainer_parameters: Dict[str, Any] = None):
        # Get type of the subclass
        model_data: Dict[str, Any] = {
            "model_type": type(self),
            "hyperparameters": self.__hyperparameters,
            "model_state_dict": self._modules.state_dict()
        }
        if trainer_parameters:
            model_data.update(trainer_parameters)
        torch.save(model_data, join(MODELS_DIR, f"{model_name}.pt"))

    def get_device(self) -> str:
        """
        Returns the device the model is running on.

        Returns:
            str: The device the model is running on.
        """
        return self._device

    def set_device(self, device: str) -> None:
        """
        Sets the device that the model is running on.

        Updates the `_device` attribute and moves the model to the new device.

        Args:
            device (str): The device to use.

        Returns:
            None
        """
        self._device = device
        self._modules.to(device)

    def train(self, mode=True) -> None:
        """
        Sets the mode of the model to train.

        Args:
            mode (bool, optional): The mode to set the model to. Defaults to True.

        Returns:
            None
        """
        self._modules.train(mode=mode)

    def eval(self) -> None:
        """
        Sets the mode of the model to eval.

        Returns:
            None
        """
        self._modules.eval()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Abstract method for the forward pass of the model. Needs to be implemented in any subclass.

        Args:
            x (Tensor): The input tensor.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
