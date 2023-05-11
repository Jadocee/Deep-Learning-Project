import logging
from abc import ABC, abstractmethod
from sys import stdout
from typing import List, Optional

from optuna import Trial, Study, create_study
from optuna import logging as optuna_logging
from optuna.trial import FrozenTrial, TrialState


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

    def run(self, study_name: Optional[str] = None, n_trials: int = 50) -> None:
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
                                    storage=f"sqlite:///studies/{study_name}.db" if study_name else None,
                                    load_if_exists=True)
        try:
            study.optimize(self._objective, n_trials=n_trials)
            pruned_trials: List[FrozenTrial] = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials: List[FrozenTrial] = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print(f"  Number of finished trials: {len(study.trials)}")
            print(f"  Number of pruned trials: {len(pruned_trials)}")
            print(f"  Number of complete trials: {len(complete_trials)}")
            print(f"Best trial:")
            trial: FrozenTrial = study.best_trial
            print(f"  Value: {trial.value}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

        except NotImplementedError:
            print("The objective function has not been implemented.")
            raise
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping the optimization.")
