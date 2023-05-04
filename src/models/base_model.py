from abc import ABC, abstractmethod

from torch.nn import Module, ModuleList


class BaseModel(ABC):
    __modules: ModuleList

    def __init__(self):
        super().__init__()
        self.__modules = ModuleList()

    def _add_modules(self, *modules: Module):
        self.__modules.extend(modules)

    @property
    def parameters(self):
        return self.__modules.parameters()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
