import csv
from itertools import chain
from typing import Tuple, Union

from nltk.lm import Vocabulary
from numpy import mean
from optuna import Trial
import pandas as pd
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from os.path import join

from models.bow_model import BOWModel
from optimisers.base_optimiser import BaseOptimiser
from trainers.bow_classifier_trainer import BOWClassifierTrainer
from utils.data_processing_utils import DataProcessingUtils
from utils.dataset_loader import DatasetLoader
from utils.definitions import STUDIES_DIR
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    vocab: Vocabulary = DataProcessingUtils.create_vocab(chain(train_data["tokens"], valid_data["tokens"]))

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

        epochs: int = trial.suggest_categorical("epochs", [5, 10, 20])
        learning_rate: float = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8, 32, 64])
        max_tokens: int = trial.suggest_categorical("max_tokens", [0,100, 200, 300, 400, 500, 600])
        optimiser_name: str = trial.suggest_categorical("optimiser", ["Adam", "RMSprop", "SGD", "Adagrad"])
    
        # Load and preprocess the data
        train_dataloader, valid_dataloader, test_dataloader, vocab = prepare_data(batch_size=batch_size,
                                                                                  max_tokens=max_tokens)

        # Create the model
        model: BOWModel = BOWModel(
            vocab_size=len(vocab),
            # embedding_dim=embedding_dim,
            # hidden_size=hidden_size,
            # n_layers=n_layers,
            # bidirectional=bidirectional,
            # pad_idx=vocab["<pad>"],
            output_size=6,
            # dropout=dropout,
        )

        # Create the trainer
        trainer: BOWClassifierTrainer = BOWClassifierTrainer(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader= test_dataloader,
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
    
    
    def validate(self,study_name): 
        output_dir: str = join(STUDIES_DIR,study_name)
        models = []
        df = pd.read_csv(join(output_dir,"results.csv"))

        sorted_df = df.sort_values(by="value",ascending=False)
        sorted_df = sorted_df.head(10)
        learning_rate = sorted_df['params_learning_rate'].astype(float)
        optimizer = sorted_df['params_optimiser']
        max_tokens = sorted_df['params_max_tokens'].astype(int)
        epochs = sorted_df['params_epochs'].astype(int)
        batch_size = sorted_df['params_batch_size'].astype(int)
        models = list(zip(learning_rate, optimizer, max_tokens, epochs, batch_size))

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
            train_dataloader, valid_dataloader, test_dataloader, vocab = prepare_data(batch_size=batch_size,
                                                                        max_tokens=tokens)
            runs = 0
            while(runs < 5):
                model: BOWModel = BOWModel(
                vocab_size=len(vocab),
                output_size=6,
                )
                runs = runs+1
                trainer: BOWClassifierTrainer = BOWClassifierTrainer(
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                vocab=vocab,
                device=self._device,
                 )
                accuracy = trainer.run(
                    model=model,
                    learning_rate =lr,
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
        top3_models.to_csv(join(output_dir,"top3.csv"), index=False)
        print(total_accs)

    def testModels(self,study_name):
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
            train_dataloader, valid_dataloader, test_dataloader, vocab = prepare_data(batch_size=batch_size,
                                                                                      max_tokens=tokens)
            model: BOWModel = BOWModel(
                vocab_size=len(vocab),
                output_size=6,
            )
            trainer: BOWClassifierTrainer = BOWClassifierTrainer(
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                vocab=vocab,
                device=self._device,
            )
            trainer.run(
                model=model,
                learning_rate=lr,
                epochs=epochs,
                optimiser_name=opt,
            )
            loss, accuracy = trainer.evaluate(model,test_dataloader)
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
        output_dir: str = join(STUDIES_DIR,studyname)

        data = pd.read_csv(join(output_dir,"results.csv"))
        # Calculate the average value for each params_optimiser group
        average_values = data.groupby('params_optimiser')['value'].mean()

        # Print the average values
        print(average_values)

    @staticmethod
    def analyseLearningRate(studyname):
        output_dir: str = join(STUDIES_DIR,studyname)

        df = pd.read_csv(join(output_dir,"results.csv"))
        df['params_learning_rate'] = pd.to_numeric(df['params_learning_rate'], errors='coerce')
        sns.scatterplot(data=df, x='params_learning_rate',
                        y='value', palette='viridis')

        # Regression line
        sns.regplot(data=df, x='params_learning_rate', y='value', scatter=False, color='red')

        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title('Learning Rate vs Accuracy')

        plt.show()
