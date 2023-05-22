from models.base_model import BaseModel


class TransformerModel(BaseModel):

    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def forward(self, x):
        # TODO: Implement me!
        raise NotImplementedError
