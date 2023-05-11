from itertools import chain
from typing import Tuple, Union, Dict, List

from nltk.lm import Vocabulary
from optuna import Trial
from torch import Tensor, stack
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from models.lstm_model import LSTMModel
from optimisers.base_optimiser import BaseOptimiser
from trainers.lstm_classifier_trainer import LSTMClassifierTrainer
from utils.data_processing_utils import DataProcessingUtils
from utils.dataset_loader import DatasetLoader


def prepare_data(batch_size: int, max_tokens: int) \
        -> Tuple[DataLoader, DataLoader, DataLoader, Union[Vocab, Vocabulary]]:
    # TODO: Move to utility class
    # TODO: Make this reusable for other datasets

    train_data, valid_data, test_data = DatasetLoader.get_tweet_topic_single_dataset()

    # Standardise the data
    train_data = train_data.map(
        lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
    valid_data = valid_data.map(
        lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
    test_data = test_data.map(
        lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})

    # Flatten the data
    flat_train = [token for tokens in train_data["tokens"] for token in tokens]
    flat_valid = [token for tokens in valid_data["tokens"] for token in tokens]
    flat_test = [token for tokens in test_data["tokens"] for token in tokens]

    # Create the vocabulary
    vocab: Vocab = DataProcessingUtils.create_vocab(chain(flat_train, flat_valid, flat_test))

    def __collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        batch_ids: Tensor = pad_sequence(([item["ids"] for item in batch]),
                                         padding_value=vocab["<pad>"], batch_first=True)
        batch_label: Tensor = stack([item["label"] for item in batch])
        batch: Dict[str, Tensor] = {"ids": batch_ids, "label": batch_label}
        return batch

    # Vectorise the data using vocabulary indexing
    train_data = train_data.map(
        lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})
    valid_data = valid_data.map(
        lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})
    test_data = test_data.map(
        lambda x: {"ids": DataProcessingUtils.vocabularise_text(tokens=x["tokens"], vocab=vocab)})

    # Convert the data to tensors
    train_data = train_data.with_format(type="torch", columns=["ids", "label"])
    valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
    test_data = test_data.with_format(type="torch", columns=["ids", "label"])

    # Create the dataloaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=__collate)
    valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch_size, collate_fn=__collate)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=__collate)

    return train_dataloader, valid_dataloader, test_dataloader, vocab


class LSTMClassifierOptimiser(BaseOptimiser):
    __vocab: Vocab
    __pad_index: int

    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def _objective(self, trial: Trial) -> float:
        # TODO: Set the seed for reproducibility

        # Suggestions for hyperparameters

        epochs: int = trial.suggest_categorical("epochs", [5, 10])
        n_layers: int = trial.suggest_int("n_layers", 1, 3)
        bidirectional: bool = trial.suggest_categorical("bidirectional", [True, False])
        learning_rate: float = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8, 32, 64, 128])
        hidden_size: int = trial.suggest_categorical("hidden_size", [128, 256, 512])
        max_tokens: int = trial.suggest_categorical("max_tokens", [100, 200, 300, 400, 500, 600])
        embedding_dim: int = trial.suggest_categorical("embedding_dim", [100, 200, 300])

        # Load and preprocess the data
        train_dataloader, valid_dataloader, test_dataloader, vocab = prepare_data(batch_size=batch_size,
                                                                                  max_tokens=max_tokens)

        # Create the model

        model: LSTMModel = LSTMModel(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            pad_idx=vocab["<pad>"],
            output_size=6,
            device=self._device,
        )

        # Create the trainer
        trainer: LSTMClassifierTrainer = LSTMClassifierTrainer(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            vocab=vocab,
            device=self._device,
        )

        # Train the model
        accuracy = trainer.run(
            model=model,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            max_tokens=max_tokens,
            trial=trial
        )

        return accuracy
