from abc import ABC, abstractmethod

from torch.nn import ModuleList


class BaseModel(ABC):
    """
    Abstract base class for models. It initializes with a device and a ModuleList, and
    provides basic model functionalities including train, eval, predict, and an abstract
    forward method.

    Attributes:
        _modules (ModuleList): A ModuleList to hold the submodules.
        _device (str): The device the model is running on, defaults to "cpu".
    """
    _modules: ModuleList
    _device: str

    def __init__(self, device: str = "cpu"):
        """
        Initializes the BaseModel class.

        Args:
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super().__init__()
        self._device = device
        self._modules = ModuleList()

    @property
    def parameters(self):
        """
        Returns the parameters of the model.

        Returns:
            Parameter: The parameters of the model.
        """
        return self._modules.parameters()

    def train(self, mode=True):
        """
        Sets the mode of the model to train.

        Args:
            mode (bool, optional): The mode to set the model to. Defaults to True.
        """
        self._modules.train(mode=mode)

    def eval(self):
        """
        Sets the mode of the model to eval.
        """
        self._modules.eval()

    def predict(self, x):
        """
        Predicts the output given the input x.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self._modules(x)

    @abstractmethod
    def forward(self, x):
        """
        Abstract method for the forward pass of the model.
        Needs to be implemented in any subclass.

        Args:
            x (torch.Tensor): The input tensor.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
