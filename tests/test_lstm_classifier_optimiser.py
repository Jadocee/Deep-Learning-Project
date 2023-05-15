from unittest import TestCase

import optuna
from optuna import Trial, create_study

from optimisers.lstm_classifier_optimiser import prepare_data


class TestLSTMClassifierOptimiser(TestCase):
    def test_objective(self):
        trial: Trial = Trial(study=create_study(), trial_id=0)
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


