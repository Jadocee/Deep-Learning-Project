from itertools import chain
from typing import Tuple, Union, Dict, List, Optional, Any

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
    """
    Prepare the data for LSTM classifier training.

    Args:
        batch_size (int): The batch size.
        max_tokens (int): The maximum number of tokens.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Union[Vocab, Vocabulary]]: Returns the train, validation, and test
        dataloaders, and the vocabulary.
    """
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

    # Create the vocabulary
    vocab: Vocab = DataProcessingUtils.create_vocab(
        chain(train_data["tokens"], valid_data["tokens"], test_data["tokens"]))

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
    """
    Class for the LSTM classifier optimiser. Inherits from the BaseOptimiser class.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initializes the LSTMClassifierOptimiser class.

        Args:
            device (str, optional): The device to use for computations. Defaults to "cpu".
        """
        super().__init__(device=device)

    def _objective(self, trial: Trial) -> float:
        """
        Objective function for the LSTM classifier optimiser. This is where the hyperparameters for the LSTM classifier
        model are suggested.

        Args:
            trial (Trial): Optuna's trial object.

        Returns:
            float: The accuracy of the model on the validation set.
        """
        # Suggestions for hyperparameters
        epochs: int = trial.suggest_categorical("epochs", [3, 5, 10, 20, 50])
        n_layers: int = trial.suggest_int("n_layers", 1, 5)
        bidirectional: bool = trial.suggest_categorical("bidirectional", [True, False])
        learning_rate: float = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8, 32, 64, 128, 256])
        hidden_size: int = trial.suggest_categorical("hidden_size", [128, 256, 512, 1024, 2048])
        max_tokens: int = trial.suggest_categorical("max_tokens", [100, 200, 300, 400, 500, 600])
        embedding_dim: int = trial.suggest_categorical("embedding_dim", [100, 200, 300])
        optimiser_name: str = trial.suggest_categorical("optimiser", ["Adam", "RMSprop", "SGD", "Adagrad"])
        dropout: float = trial.suggest_float("dropout", 0.0, 0.5)
        lr_scheduler_name: Optional[str] = trial.suggest_categorical("lr_scheduler",
                                                                     ["StepLR", "ExponentialLR", "MultiStepLR", None,
                                                                      "ReduceLROnPlateau", "CosineAnnealingLR"])
        # Suggest hyperparameters for the learning rate scheduler based on the chosen learning rate scheduler
        kwargs: Dict[str, Any] = dict()
        if lr_scheduler_name == "StepLR":
            step_size: int = trial.suggest_int("step_size", 1, 100)
            gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
            kwargs.update({"step_size": step_size, "gamma": gamma})
        elif lr_scheduler_name == "ExponentialLR":
            gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
            kwargs.update({"gamma": gamma})
        elif lr_scheduler_name == "MultiStepLR":
            n_milestones: int = trial.suggest_int("n_milestones", 1, 5)
            milestones: List[int] = list()
            for i in range(n_milestones):
                milestones.append(trial.suggest_int(f"milestone_{i}", 1, epochs - 1))
            gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
            kwargs.update({"milestones": milestones, "gamma": gamma})
        elif lr_scheduler_name == "ReduceLROnPlateau":
            factor: float = trial.suggest_float("factor", 0.1, 0.9)
            patience: int = trial.suggest_int("patience", 1, 10)
            threshold: float = trial.suggest_float("threshold", 1e-4, 1e-1)
            threshold_mode: str = trial.suggest_categorical("threshold_mode", ["rel", "abs"])
            cooldown: int = trial.suggest_int("cooldown", 1, 10)
            min_lr: float = trial.suggest_float("min_lr", 1e-6, 1e-1, log=True)
            kwargs.update({"factor": factor, "patience": patience, "threshold": threshold,
                           "threshold_mode": threshold_mode, "cooldown": cooldown, "min_lr": min_lr})
        elif lr_scheduler_name == "CosineAnnealingLR":
            T_max: int = trial.suggest_int("T_max", 1, epochs - 1)
            eta_min: float = trial.suggest_float("eta_min", 1e-4, 1e-1)
            kwargs.update({"T_max": T_max, "eta_min": eta_min})

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
            dropout=dropout,
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
            trial=trial,
            optimiser_name=optimiser_name,
            lr_scheduler_name=lr_scheduler_name,
            kwargs=kwargs
        )

        return accuracy
