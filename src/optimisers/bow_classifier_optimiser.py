import csv
from itertools import chain
from typing import Any, Dict, Optional, Tuple, Union

from nltk.lm import Vocabulary
from numpy import mean
from optuna import Trial
import pandas as pd
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from os.path import join
from datasets import DatasetDict, Dataset

from models.bow_model import BOWModel
from optimisers.base_optimiser import BaseOptimiser
from trainers.bow_classifier_trainer import BOWClassifierTrainer
from utils.dataset_loader import DatasetLoader
from utils.definitions import STUDIES_DIR
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.results_utils import ResultsUtils

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
        scheduler_hyperparams: Optional[Dict[str, Any]] = self._define_scheduler_hyperparams(trial)

        # Load and preprocess the data
        train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                  max_tokens=max_tokens)

        # Create the model
        model: BOWModel = BOWModel(
            vocab_size=len(self.__vocab),
            output_size=6,
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


    def validate(self, study_name):
        output_dir: str = join(STUDIES_DIR, study_name)
        models = []
        df = pd.read_csv(join(output_dir, "results.csv"))

        sorted_df = df.sort_values(by="value", ascending=False)
        sorted_df = sorted_df.head(10)
        learning_rate = sorted_df['params_learning_rate'].astype(float)
        optimizer = sorted_df['params_optimiser']
        max_tokens = sorted_df['params_max_tokens'].astype(int)
        epochs = sorted_df['params_epochs'].astype(int)
        batch_size = sorted_df['params_batch_size'].astype(int)
        models = list(zip(learning_rate, optimizer,
                      max_tokens, epochs, batch_size))

        models = pd.DataFrame({
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'max_tokens': max_tokens,
            'epochs': epochs,
            'batch_size': batch_size
        })

        models['average_accuracy'] = None
        total_accs = []
        for i, row in models.iterrows():
            lr = row['learning_rate']
            opt = row['optimizer']
            tokens = row['max_tokens']
            epochs = row['epochs']
            batch_size = row['batch_size']
            accuracies = []
            train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                      max_tokens=tokens)
            runs = 0
            while(runs < 5):
                model: BOWModel = BOWModel(
                    vocab_size=len(self.__vocab),
                    output_size=6,
                )
                runs = runs+1
                trainer: BOWClassifierTrainer = BOWClassifierTrainer(
                    train_dataloader=train_dataloader,
                    valid_dataloader=valid_dataloader,
                    test_dataloader=test_dataloader,
                    vocab=self.__vocab,
                    device=self._device,
                )
                accuracy = trainer.run(
                    model=model,
                    learning_rate=lr,
                    epochs=epochs,
                    optimiser_name=opt,
                )
                accuracies.append(accuracy)
            average_accuracy = mean(accuracies)
            total_accs.append(average_accuracy)
            models.loc[i, 'average_accuracy'] = average_accuracy
        print(models)
        models['average_accuracy'] = pd.to_numeric(models['average_accuracy'])
        top3_models = models.nlargest(3, 'average_accuracy')
        top3_models.to_csv(join(output_dir, "top3.csv"), index=False)
        print(total_accs)

    def testModels(self, study_name):
        output_dir: str = join(STUDIES_DIR, study_name)
        models = []
        df = pd.read_csv(join(output_dir, "top3.csv"))

        learning_rate = df['learning_rate'].astype(float)
        optimizer = df['optimizer']
        max_tokens = df['max_tokens'].astype(int)
        epochs = df['epochs'].astype(int)
        batch_size = df['batch_size'].astype(int)
        models = list(zip(learning_rate, optimizer,
                      max_tokens, epochs, batch_size))

        models = pd.DataFrame({
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'max_tokens': max_tokens,
            'epochs': epochs,
            'batch_size': batch_size
        })

        models['Accuracy'] = None
        models['Loss'] = None
        for i, row in models.iterrows():
            lr = row['learning_rate']
            opt = row['optimizer']
            tokens = row['max_tokens']
            epochs = row['epochs']
            batch_size = row['batch_size']
            train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                      max_tokens=tokens)
            model: BOWModel = BOWModel(
                vocab_size=len(self.__vocab),
                output_size=6,
            )
            trainer: BOWClassifierTrainer = BOWClassifierTrainer(
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                vocab=self.__vocab,
                device=self._device,
            )
            trainer.run(
                model=model,
                learning_rate=lr,
                epochs=epochs,
                optimiser_name=opt,
            )
            loss, accuracy = trainer.evaluate(model, test_dataloader)
            print(accuracy)
            print(loss)
            models.loc[i, 'Accuracy'] = accuracy
            models.loc[i, 'Loss'] = loss
        print(models)
        models['Accuracy'] = pd.to_numeric(models['Accuracy'])
        models['Loss'] = pd.to_numeric(models['Loss'])
        models.to_csv(join(output_dir, "TestResults.csv"), index=False)

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
