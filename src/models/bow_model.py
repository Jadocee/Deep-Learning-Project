from torch import Tensor
import torch
import torch.nn as nn
from models.base_model import BaseModel


class BOWModel(BaseModel):
    
    def __init__(self, vocab_size: int,output_size,n_layers: int, layer_size:int, device: str = "cpu"):
        super().__init__(device=device)
        self._modules.append(nn.Linear(vocab_size, layer_size))
        for layer in range(n_layers):
            self._modules.append(nn.ReLU())
            self._modules.append(nn.Linear(layer_size, layer_size))

        self._modules.append(nn.ReLU())
        self._modules.append(nn.Linear(layer_size,output_size))
        self._modules.to(self._device)
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules:
            # Cast tensor to a different type using to()
            x = x.to(dtype=torch.float32)  # Cast to float32
            x = module(x)
        return x
    