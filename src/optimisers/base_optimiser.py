import logging
from abc import ABC, abstractmethod
from sys import stdout
from typing import List, Optional

from optuna import Trial, Study, create_study
from optuna import logging as optuna_logging
from optuna.pruners import MedianPruner, NopPruner
from optuna.trial import FrozenTrial, TrialState
from pandas import DataFrame

from utils.definitions import STUDIES_DIR


class BaseOptimiser(ABC):
    _device: str

    # TODO: Add dictionaries for storing the best hyperparameters and the best model.

    def __init__(self, device: str = "cpu") -> None:

        super().__init__()
        self._device = device

    @abstractmethod
    def _objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna hyperparameter optimization.
        :param trial: Trial object.
        :return: Objective function value.
        """
        raise NotImplementedError

    def run(self, study_name: Optional[str] = None, n_trials: int = 50, prune: bool = True) -> None:
        """
        Run the hyperparameter optimization.
        :param study_name:
        :param n_trials: The number of trials to perform. Default: 50.
        :return: None.
        """

        optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(stdout))

        # Create the study
        study: Study = create_study(direction="maximize",
                                    study_name=study_name,
                                    pruner=MedianPruner() if prune else NopPruner())
        try:
            study.optimize(self._objective, n_trials=n_trials)
            pruned_trials: List[FrozenTrial] = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials: List[FrozenTrial] = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Summary: ")
            print(f"\tNumber of finished trials: {len(study.trials)}")
            print(f"\tNumber of pruned trials: {len(pruned_trials)}")
            print(f"\tNumber of complete trials: {len(complete_trials)}")
            print(f"Best trial:")
            trial: FrozenTrial = study.best_trial
            print(f"\tValue: {trial.value}")
            print("Params: ")
            for key, value in trial.params.items():
                print(f"\t{key}: {value}")

            df: DataFrame = study.trials_dataframe()
            out_csv: str = f"{STUDIES_DIR}/{study.study_name}.csv"
            df.to_csv(out_csv, index=False)
            print(f"Saved results to {out_csv}")

        except NotImplementedError:
            print("The objective function has not been implemented.")
            raise
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping the optimization.")
