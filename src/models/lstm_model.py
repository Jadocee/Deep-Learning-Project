from torch import Tensor, amax
from torch.nn import Embedding, LSTM, Linear

from models.base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, vocab_size: int, embedding_dim: int, input_size: int, output_size: int, hidden_size: int,
                 n_layers: int, pad_idx: int, bidirectional: bool = True):
        super().__init__()
        self._add_modules(
            Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx),
            LSTM(input_size=input_size, hidden_size=n_layers, num_layers=n_layers, bidirectional=bidirectional),
            Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.__modules[0](x)
        x, _ = self.__modules[1](x)
        x = amax(x, dim=1)
        x = self.__modules[2](x)
        return x
