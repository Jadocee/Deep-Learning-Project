from torch import Tensor, amax
from torch.nn import Embedding, LSTM, Linear

from models.base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, vocab_size: int, embedding_dim: int, output_size: int, hidden_size: int, n_layers: int,
                 pad_idx: int, bidirectional: bool = True, device: str = "cpu"):
        """
        LSTM model for text classification.
        :param vocab_size: The size of the vocabulary.
        :param embedding_dim: The dimension of the word embeddings and the size of the input feature vector (the input
        to the LSTM layer is the output of the Embedding layer).
        :param output_size: The size of the output feature vector (number of classes).
        :param hidden_size: The number of hidden units in the LSTM layers.
        :param n_layers: The number of layers in the LSTM.
        :param pad_idx: The index of the padding token in the vocabulary.
        :param bidirectional: Whether to use a bidirectional LSTM. If a bidirectional LSTM is used, the output size of
        the LSTM layer is doubled, thus, the input size of the Linear layer is also doubled.
        """
        super().__init__(device=device)
        self._modules.extend([
            Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx).to(self._device),
            LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers,
                 bidirectional=bidirectional).to(self._device),
            Linear(in_features=hidden_size * 2 if bidirectional else hidden_size, out_features=output_size).to(
                self._device)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        :param x: The input tensor, of shape (batch_size, max_tokens).
        :return: The output tensor, of shape (batch_size, output_size).
        """
        x = self._modules[0](x)
        x, _ = self._modules[1](x)
        x = amax(x, dim=1)
        x = self._modules[2](x)
        return x
