from os.path import join
from typing import Tuple, Dict, List, Optional, Any

from datasets import DatasetDict, Dataset
from optuna import Trial
from torch import Tensor, stack
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from models.lstm_model import LSTMModel
from optimisers.base_optimiser import BaseOptimiser
from trainers.lstm_classifier_trainer import LSTMClassifierTrainer
from utils.dataset_loader import DatasetLoader
from utils.definitions import STUDIES_DIR
from utils.results_utils import ResultsUtils
from utils.text_preprocessor import TextPreprocessor


class LSTMClassifierOptimiser(BaseOptimiser):
    """
    Class for the LSTM classifier optimiser. Inherits from the BaseOptimiser class.

    Attributes:
        __vocab (Vocab): A torchtext Vocab object which encodes the vocabulary used in the dataset.
        __max_tokens (int): The maximum number of tokens to pass to the LSTM model. The tokens are truncated
            if they exceed this value during the standardisation process.
    """

    __vocab: Vocab

    def __init__(self, device: str = "cpu") -> None:
        """
        Initializes the LSTMClassifierOptimiser class.

        Args:
            device (str, optional): The device to use for computations. Defaults to "cpu".

        Notes:
            - A lower value for `max_tokens` will result less memory usage, but may result in a loss of information.
            - A larger value for `max_tokens` will result in more memory usage, but may result in a more accurate model.

        Returns:
            None
        """
        super().__init__(device=device)

    def __collate_batch(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Collates a batch of data into a dictionary.

        Args:
            batch (List[Dict[str, Tensor]]): A list of dictionaries containing the data to collate.

        Returns:
            Dict[str, Tensor]: A dictionary containing the collated data.
        """

        batch_ids: Tensor = pad_sequence(([item["ids"] for item in batch]),
                                         padding_value=self.__vocab["<pad>"], batch_first=True)
        batch_label: Tensor = stack([item["label"] for item in batch])
        batch: Dict[str, Tensor] = {"ids": batch_ids, "label": batch_label}
        return batch

    def _prepare_data(self, batch_size: int, max_tokens: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Performs the data preparation step for the LSTM classifier optimiser.

        Prepares the data by loading the training, validation, and testing splits of the Tweet Topic dataset,
        standardising and tokenising the text, and creating a vocabulary from the training data. The data is then
        split into batches and loaded into DataLoaders.

        Notes:
            - The DataLoaders use the `_collate` method to collate the data into batches.
            - The DataLoaders are stored in the `_train_dataloader`, `_valid_dataloader`, and `_test_dataloader`
                attributes.
            - The vocabulary is stored in the `_vocab` attribute.
            - The maximum number of tokens is stored in the `_max_tokens` attribute.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: The training, validation, and testing DataLoaders, respectively.
        """

        splits: DatasetDict[str, Dataset] = DatasetLoader.get_tweet_topic_single_dataset()
        preprocessor: TextPreprocessor = TextPreprocessor(max_tokens=max_tokens)
        splits = preprocessor.preprocess_dataset_dict(dataset_dict=splits)
        self.__vocab = preprocessor.get_vocab()
        train_dataloader = DataLoader(dataset=splits["train"], shuffle=True, collate_fn=self.__collate_batch,
                                      batch_size=batch_size)
        valid_dataloader = DataLoader(dataset=splits["validation"], shuffle=True, collate_fn=self.__collate_batch,
                                      batch_size=batch_size)
        test_dataloader = DataLoader(dataset=splits["test"], collate_fn=self.__collate_batch,
                                     batch_size=batch_size)

        return train_dataloader, valid_dataloader, test_dataloader

    def _objective(self, trial: Trial) -> float:
        """
        Objective function for the LSTM classifier optimiser. This is where the hyperparameters for the LSTM classifier
        model are suggested.

        Args:
            trial (Trial): The trial object for the current Optuna study.

        Returns:
            float: The accuracy of the model on the validation set.
        """
        self._logger.info(f"Trial number: {trial.number}")

        epochs: int = trial.suggest_categorical("epochs", [5, 10])
        learning_rate: float = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        optimiser_name: str = trial.suggest_categorical("optimiser", ["Adam"])
        n_layers: int = trial.suggest_int("n_layers", 1, 5)
        bidirectional: bool = trial.suggest_categorical("bidirectional", [True, False])
        hidden_size: int = trial.suggest_categorical("hidden_size", [256, 512, 1024, 2048])
        embedding_dim: int = trial.suggest_categorical("embedding_dim", [100, 200, 300, 400, 500])
        if n_layers > 1:
            dropout: float = trial.suggest_float("dropout", 0.1, 0.5)
        else:
            dropout: float = trial.suggest_float("dropout", 0.0, 0.0)
        scheduler_hyperparams: Optional[Dict[str, Any]] = self._define_scheduler_hyperparams(trial)

        self._logger.info(f"Selected hyperparameters: {trial.params.__str__()}")

        train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size, max_tokens=1000)
        model: LSTMModel = LSTMModel(
            vocab_size=len(self.__vocab),
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            output_size=6,
            pad_idx=self.__vocab["<pad>"],
            device=self._device
        )
        trainer = LSTMClassifierTrainer(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            vocab=self.__vocab,
            device=self._device
        )
        results: Dict[str, Any] = trainer.fit(
            model=model,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            trial=trial,
            optimiser_name=optimiser_name,
            lr_scheduler_params=scheduler_hyperparams
        )
        save_path: str = join(STUDIES_DIR, trial.study.study_name, f"trial_{trial.number}_{model.get_id()}")
        trial.set_user_attr(key="save_path", value=save_path)
        ResultsUtils.plot_loss_and_accuracy_curves(
            training_losses=results["train_losses"],
            validation_losses=results["valid_losses"],
            training_accuracies=results["train_accuracies"],
            validation_accuracies=results["valid_accuracies"],
            save_path=save_path
        )
        ResultsUtils.plot_confusion_matrix(cm=results["confusion_matrix"], save_path=save_path)

        return results["valid_accuracies"][-1]
