from torch import Tensor
import torch
import torch.nn as nn
from models.base_model import BaseModel


class BOWModel(BaseModel):
    
    def __init__(self, vocab_size: int,output_size):
        super().__init__()
        self._modules.append(nn.Linear(vocab_size, 12))
        self._modules.append(nn.ReLU())
        self._modules.append(nn.Linear(12, 16))
        self._modules.append(nn.ReLU())
        self._modules.append(nn.Linear(16,output_size))
        self._modules.to(self._device)
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules:
            # Cast tensor to a different type using to()
            x = x.to(dtype=torch.float32)  # Cast to float32
            x = module(x)
        return x
    