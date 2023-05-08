from typing import Tuple

from nltk.lm import Vocabulary
from optuna import Trial
from torch.utils.data import DataLoader

from models.lstm_model import LSTMModel
from optimisers.base_optimiser import BaseOptimiser
from trainers.lstm_classifier_trainer import LSTMClassifierTrainer
from utils.data_processing_utils import DataProcessingUtils
from utils.dataset_loader import DatasetLoader
from utils.definitions import TWEET_TOPIC_SINGLE, TWEET_TOPIC_SINGLE_TRAIN_SPLIT, TWEET_TOPIC_SINGLE_TEST_SPLIT


def prepare_data(batch_size: int, max_tokens: int) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    # TODO: Move to utility class
    # TODO: Make this reusable for other datasets

    train_data, valid_data, test_data = DatasetLoader.get_dataset(
        dataset_name=TWEET_TOPIC_SINGLE,
        train_split=TWEET_TOPIC_SINGLE_TRAIN_SPLIT,
        test_split=TWEET_TOPIC_SINGLE_TEST_SPLIT
    )

    # Standardise the data
    train_data = train_data.map(
        lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
    valid_data = valid_data.map(
        lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
    test_data = test_data.map(
        lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})

    # Create the vocabulary
    vocab: Vocabulary = DataProcessingUtils.create_vocab_1(word_tokens=train_data["tokens"])

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
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader, vocab


class LSTMClassifierOptimiser(BaseOptimiser):

    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def _objective(self, trial: Trial) -> float:
        # TODO: Set the seed for reproducibility

        # Suggestions for hyperparameters
        learning_rate: float = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8, 32, 64, 128])
        hidden_size: int = trial.suggest_categorical("hidden_size", [128, 256, 512])
        max_tokens: int = trial.suggest_categorical("max_tokens", [100, 200, 300, 400, 500, 600])
        embedding_dim: int = trial.suggest_categorical("embedding_dim", [100, 200, 300])
        n_layers: int = trial.suggest_int("n_layers", 1, 3)
        bidirectional: bool = trial.suggest_categorical("bidirectional", [True, False])

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
            output_size=len(train_dataloader.dataset["label"])
        )

        # Create the trainer
        trainer: LSTMClassifierTrainer = LSTMClassifierTrainer(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            device=self._device,
        )

        # Train the model
        accuracy: float = trainer.run(
            model=model,
            learning_rate=learning_rate,
            epochs=5,
            batch_size=batch_size,
            max_tokens=max_tokens,
            test=False
        )

        return accuracy
