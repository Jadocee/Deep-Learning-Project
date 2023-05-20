import torch.nn as nn


class BOWModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.hidden = nn.Linear(vocab_size, 12)
        self.hidden2 = nn.Linear(12, 16)
        self.out = nn.Linear(16, 6)

    def forward(self, x):
        x = nn.ReLU()(self.hidden(x))
        x = nn.ReLU()(self.hidden2(x))
        x = self.out(x)
        return x
