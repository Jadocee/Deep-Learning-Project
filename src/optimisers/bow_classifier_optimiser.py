from typing import Tuple, Union

from nltk.lm import Vocabulary
from optuna import Trial
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from models.bow_model import BOWModel
from optimisers.base_optimiser import BaseOptimiser
from trainers.bow_classifier_trainer import BOWClassifierTrainer
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

    # # Create the vocabulary
    vocab: Vocabulary = DataProcessingUtils.create_vocab(train_data, valid_data)

    # Numericalize the data
    train_data = train_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})
    valid_data = valid_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})
    test_data = test_data.map(DataProcessingUtils.numericalize_data, fn_kwargs={'vocab': vocab})

    train_data = train_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
    valid_data = valid_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
    test_data = test_data.map(DataProcessingUtils.multi_hot_data, fn_kwargs={'num_classes': len(vocab)})

    # Convert the data to tensors
    train_data = train_data.with_format(type="torch", columns=["multi_hot", "label"])
    valid_data = valid_data.with_format(type="torch", columns=["multi_hot", "label"])
    test_data = test_data.with_format(type="torch", columns=["multi_hot", "label"])

    # Create the dataloaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader, vocab


class BOWClassifierOptimiser(BaseOptimiser):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def _objective(self, trial: Trial) -> float:
        # TODO: Set the seed for reproducibility

        # Suggestions for hyperparameters

        epochs: int = trial.suggest_categorical("epochs", [5, 10, 20, 50,100])
        learning_rate: float = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8, 32, 64, 128, 256])
        max_tokens: int = trial.suggest_categorical("max_tokens", [100, 200, 300, 400, 500, 600])
        optimiser_name: str = trial.suggest_categorical("optimiser", ["Adam", "RMSprop", "SGD", "Adagrad"])
        # lr_scheduler_name: Optional[str] = trial.suggest_categorical("lr_scheduler",
        #                                                              ["StepLR", "ExponentialLR", "MultiStepLR", None,
        #                                                               "ReduceLROnPlateau", "CosineAnnealingLR"])
        #
        # # Suggest hyperparameters for the learning rate scheduler based on the chosen learning rate scheduler
        # kwargs: Dict[str, Any] = dict()
        # if lr_scheduler_name == "StepLR":
        #     step_size: int = trial.suggest_int("step_size", 1, 100)
        #     gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
        #     kwargs.update({"step_size": step_size, "gamma": gamma})
        # elif lr_scheduler_name == "ExponentialLR":
        #     gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
        #     kwargs.update({"gamma": gamma})
        # elif lr_scheduler_name == "MultiStepLR":
        #     n_milestones: int = trial.suggest_int("n_milestones", 1, 5)
        #     milestones: List[int] = list()
        #     for i in range(n_milestones):
        #         milestones.append(trial.suggest_int(
        #             f"milestone_{i}", 1, epochs - 1))
        #     gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
        #     kwargs.update({"milestones": milestones, "gamma": gamma})
        # elif lr_scheduler_name == "ReduceLROnPlateau":
        #     factor: float = trial.suggest_float("factor", 0.1, 0.9)
        #     patience: int = trial.suggest_int("patience", 1, 10)
        #     threshold: float = trial.suggest_float("threshold", 1e-4, 1e-1)
        #     threshold_mode: str = trial.suggest_categorical(
        #         "threshold_mode", ["rel", "abs"])
        #     cooldown: int = trial.suggest_int("cooldown", 1, 10)
        #     min_lr: float = trial.suggest_float("min_lr", 1e-6, 1e-1, log=True)
        #     kwargs.update({"factor": factor, "patience": patience, "threshold": threshold,
        #                    "threshold_mode": threshold_mode, "cooldown": cooldown, "min_lr": min_lr})
        # elif lr_scheduler_name == "CosineAnnealingLR":
        #     T_max: int = trial.suggest_int("T_max", 1, epochs - 1)
        #     eta_min: float = trial.suggest_float("eta_min", 1e-4, 1e-1)
        #     kwargs.update({"T_max": T_max, "eta_min": eta_min})

        # Load and preprocess the data
        train_dataloader, valid_dataloader, test_dataloader, vocab = prepare_data(batch_size=batch_size,
                                                                                  max_tokens=max_tokens)

        # Create the model
        model: BOWModel = BOWModel(
            vocab_size=len(vocab),
            output_size=6,
        )

        # Create the trainer
        trainer: BOWClassifierTrainer = BOWClassifierTrainer(
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
            optimiser_name=optimiser_name,
            # lr_scheduler_name=lr_scheduler_name,
            # kwargs=kwargs
        )

        return accuracy
