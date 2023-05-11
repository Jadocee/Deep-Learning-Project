from abc import ABC, abstractmethod

from torch.nn import ModuleList


class BaseModel(ABC):
    _modules: ModuleList
    _device: str

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self._device = device
        self._modules = ModuleList().to(device)

    @property
    def parameters(self):
        return self._modules.parameters()

    def train(self, mode=True):
        self._modules.train(mode=mode)

    def eval(self):
        self._modules.eval()

    def predict(self, x):
        return self._modules(x)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
