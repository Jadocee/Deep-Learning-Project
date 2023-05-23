from typing import Tuple

from optuna import Trial
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from datasets import DatasetDict
from models.bow_model import BOWModel
from optimisers.base_optimiser import BaseOptimiser
from trainers.bow_classifier_trainer import BOWClassifierTrainer
from utils.dataset_loader import DatasetLoader
from utils.text_preprocessor import TextPreprocessor


class BOWClassifierOptimiser(BaseOptimiser):
    __vocab: Vocab

    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def _prepare_data(self, batch_size: int, max_tokens: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # TODO: Move to utility class
        # TODO: Make this reusable for other datasets

        dataset_dict: DatasetDict = DatasetLoader.get_tweet_topic_single_dataset()
        preprocessor: TextPreprocessor = TextPreprocessor(encode=True, encoding_method="multi-hot",
                                                          max_tokens=max_tokens)
        dataset_dict = preprocessor.preprocess_dataset_dict(dataset_dict=dataset_dict)
        self.__vocab = preprocessor.get_vocab()
        train_dataloader = DataLoader(dataset=dataset_dict["train"], batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset=dataset_dict["validation"], shuffle=True, batch_size=batch_size)
        test_dataloader = DataLoader(dataset=dataset_dict["test"], batch_size=batch_size)

        # Standardise the data
        # train_data = train_data.map(
        #     lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        # valid_data = valid_data.map(
        #     lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        # test_data = test_data.map(
        #     lambda x: {"tokens": DataProcessingUtils.standardise_text(text=x["text"], max_tokens=max_tokens)})
        #
        # # # Create the vocabulary
        # vocab: Vocabulary = DataProcessingUtils.create_vocab_2(train_data, valid_data, test_data)
        #
        # # Numericalize the data
        # train_data = train_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})
        # valid_data = valid_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})
        # test_data = test_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})
        #
        # train_data = train_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
        # valid_data = valid_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
        # test_data = test_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
        #
        # # Convert the data to tensors
        # train_data = train_data.with_format(type="torch", columns=["multi_hot", "label"])
        # valid_data = valid_data.with_format(type="torch", columns=["multi_hot", "label"])
        # test_data = test_data.with_format(type="torch", columns=["multi_hot", "label"])

        # Create the dataloaders
        # train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        # valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch_size)
        # test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)

        return train_dataloader, valid_dataloader, test_dataloader

    def _objective(self, trial: Trial) -> float:
        # Suggestions for hyperparameters

        epochs: int = trial.suggest_categorical("epochs", [5, 10, 20])
        learning_rate: float = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8, 32, 64, 128, 256])
        max_tokens: int = trial.suggest_categorical("max_tokens", [100, 200, 300, 400, 500, 600])
        optimiser_name: str = trial.suggest_categorical("optimiser", ["Adam", "RMSprop", "SGD", "Adagrad"])

        # Load and preprocess the data
        train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                 max_tokens=max_tokens)

        # Create the model
        model: BOWModel = BOWModel(
            vocab_size=len(self.__vocab),
            # embedding_dim=embedding_dim,
            # hidden_size=hidden_size,
            # n_layers=n_layers,
            # bidirectional=bidirectional,
            # pad_idx=vocab["<pad>"],
            # output_size=6,
            # dropout=dropout,
        )

        # Create the trainer
        trainer: BOWClassifierTrainer = BOWClassifierTrainer(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            vocab=self.__vocab,
            device=self._device,
        )

        # Train the model
        accuracy = trainer.fit(
            model=model,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            max_tokens=max_tokens,
            trial=trial,
            optimiser_name=optimiser_name,
            # lr_scheduler_name=lr_scheduler_name,
            # kwargs=kwargs
        )

        return accuracy
