from torch import Tensor, amax
from torch.nn import Embedding, LSTM, Linear

from models.base_model import BaseModel


class LSTMModel(BaseModel):
    """
    An LSTM model for text classification, inheriting from the base model. This model uses embedding, LSTM,
    and linear layers for its operations.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, output_size: int, hidden_size: int, n_layers: int,
                 pad_idx: int, dropout: float = 0.0, bidirectional: bool = True, device: str = "cpu"):
        """
        Initializes the LSTMModel class with given parameters.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of the word embeddings and the size of the input feature vector.
            output_size (int): The size of the output feature vector (number of classes).
            hidden_size (int): The number of hidden units in the LSTM layers.
            n_layers (int): The number of layers in the LSTM.
            pad_idx (int): The index of the padding token in the vocabulary.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            bidirectional (bool, optional): Whether to use a bidirectional LSTM. Defaults to True.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super().__init__(device=device)
        self._modules \
            .extend([
            Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx),
            LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers,
                 bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0),
            Linear(in_features=hidden_size * 2 if bidirectional else hidden_size, out_features=output_size)
        ]) \
            .to(self._device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the LSTMModel.

        Args:
            x (Tensor): The input tensor, of shape (batch_size, max_tokens).

        Returns:
            Tensor: The output tensor, of shape (batch_size, output_size).
        """
        # Pass the input through the embedding layer.
        x = self._modules[0](x)
        # Pass the output of the embedding layer through the LSTM layer.
        x, _ = self._modules[1](x)
        # Get the maximum value over the sequence dimension (dim=1) for each batch element.
        x = amax(x, dim=1)
        # Pass the output of the LSTM layer through the Linear layer.
        x = self._modules[2](x)
        return x
