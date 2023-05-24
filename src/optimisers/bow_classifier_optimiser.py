from typing import Any, Dict, Optional, Tuple

from optuna import Trial
import pandas as pd
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from os.path import join

from datasets import DatasetDict
from models.bow_model import BOWModel
from optimisers.base_optimiser import BaseOptimiser
from trainers.bow_classifier_trainer import BOWClassifierTrainer
from utils.dataset_loader import DatasetLoader
from utils.definitions import STUDIES_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from utils.results_utils import ResultsUtils
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from utils.text_preprocessor import TextPreprocessor


class BOWClassifierOptimiser(BaseOptimiser):
    __vocab: Vocab

    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def _prepare_data(self, batch_size: int, max_tokens: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # TODO: Move to utility class
        # TODO: Make this reusable for other custom_datasets

        dataset_dict: DatasetDict = DatasetLoader.get_tweet_topic_single_dataset()
        preprocessor: TextPreprocessor = TextPreprocessor(encode=True, encoding_method="multi-hot",
                                                          max_tokens=max_tokens)
        dataset_dict = preprocessor.preprocess_dataset_dict(
            dataset_dict=dataset_dict)
        self.__vocab = preprocessor.get_vocab()
        train_dataloader = DataLoader(
            dataset=dataset_dict["train"], batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(
            dataset=dataset_dict["validation"], shuffle=True, batch_size=batch_size)
        test_dataloader = DataLoader(
            dataset=dataset_dict["test"], batch_size=batch_size)

        return train_dataloader, valid_dataloader, test_dataloader

    def _objective(self, trial: Trial) -> float:
        # Suggestions for hyperparameters
        epochs: int = trial.suggest_categorical("epochs", [5, 10, 20])
        learning_rate: float = trial.suggest_float(
            "learning_rate", 1e-5, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8, 32, 64])
        max_tokens: int = trial.suggest_categorical(
            "max_tokens", [100, 200, 300, 400, 500, 600])
        optimiser_name: str = trial.suggest_categorical(
            "optimiser", ["Adam", "RMSprop", "SGD", "Adagrad"])
        hidden_size: int = trial.suggest_categorical("hidden_size", [64, 128, 256])
        n_layers: int = trial.suggest_categorical("n_layers", [2, 4, 6])
        scheduler_hyperparams: Optional[Dict[str, Any]] = self._define_scheduler_hyperparams(trial)

        # Load and preprocess the data
        train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                 max_tokens=max_tokens)

        # Create the model
        model: BOWModel = BOWModel(
            vocab_size=len(self.__vocab),
            output_size=6,
            n_layers=n_layers,
            layer_size=hidden_size,
            device=self._device
        )

        # Create the trainer
        trainer: BOWClassifierTrainer = BOWClassifierTrainer(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            vocab=self.__vocab,
            device=self._device,
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
        ResultsUtils.plot_loss_and_accuracy_curves(
            training_losses=results["train_losses"],
            validation_losses=results["valid_losses"],
            training_accuracies=results["train_accuracies"],
            validation_accuracies=results["valid_accuracies"],
            save_path=save_path
        )
        ResultsUtils.plot_confusion_matrix(cm=results["confusion_matrix"], save_path=save_path)

        return results["valid_accuracies"][-1]

    def _evaluate_test(self, trial: Trial, save_path, number):
        # Suggestions for hyperparameters
        epochs: int = trial.suggest_categorical("epochs", [5, 10, 20])
        learning_rate: float = trial.suggest_float(
            "learning_rate", 1e-5, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8, 32, 64])
        max_tokens: int = trial.suggest_categorical(
            "max_tokens", [100, 200, 300, 400, 500, 600])
        optimiser_name: str = trial.suggest_categorical(
            "optimiser", ["Adam", "RMSprop", "SGD", "Adagrad"])
        hidden_size: int = trial.suggest_categorical("hidden_size", [64, 128, 256])
        n_layers: int = trial.suggest_categorical("n_layers", [2, 4, 6])
        scheduler_hyperparams: Optional[Dict[str, Any]] = self._define_scheduler_hyperparams(trial)

        # Load and preprocess the data
        train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                 max_tokens=max_tokens)

        # Create the model
        model: BOWModel = BOWModel(
            vocab_size=len(self.__vocab),
            output_size=6,
            n_layers=n_layers,
            layer_size=hidden_size,
            device=self._device
        )

        # Create the trainer
        trainer: BOWClassifierTrainer = BOWClassifierTrainer(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            vocab=self.__vocab,
            device=self._device,
        )

        trainer.fit(
            model=model,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            trial=trial,
            optimiser_name=optimiser_name,
            lr_scheduler_params=scheduler_hyperparams
        )

        loss, accuracy, preds, targets = trainer._evaluate(
            model=model,
            dataloader=test_dataloader
        )

        ResultsUtils.record_performance_scores(
            scores={
                "accuracy": accuracy_score(y_true=targets, y_pred=preds),
                "precision": precision_score(y_true=targets, y_pred=preds, average="macro"),
                "f1": f1_score(y_true=targets, y_pred=preds, average="macro"),
                "recall": recall_score(y_true=targets, y_pred=preds, average="macro"),
            },
            save_path=save_path,
            appendName=number
        )

        ResultsUtils.plot_confusion_matrix(
            cm=confusion_matrix(y_true=targets, y_pred=preds),
            save_path=save_path,
            appendName=number
        )
        print(f"Accuracy:{accuracy}       Loss:{loss}")
        return accuracy, loss, preds, targets

    @staticmethod
    def analyseOptimizerImpact(studyname):
        output_dir: str = join(STUDIES_DIR, studyname)

        data = pd.read_csv(join(output_dir, "results.csv"))
        # Calculate the average value for each params_optimiser group
        average_values = data.groupby('params_optimiser')['value'].mean()

        # Print the average values
        print(average_values)

    @staticmethod
    def analyseLearningRate(studyname):
        output_dir: str = join(STUDIES_DIR, studyname)

        df = pd.read_csv(join(output_dir, "results.csv"))
        df['params_learning_rate'] = pd.to_numeric(
            df['params_learning_rate'], errors='coerce')
        sns.scatterplot(data=df, x='params_learning_rate',
                        y='value', palette='viridis')
        # Regression line
        sns.regplot(data=df, x='params_learning_rate',
                    y='value', scatter=False, color='red')

        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title('Learning Rate vs Accuracy')
        plt.show()
