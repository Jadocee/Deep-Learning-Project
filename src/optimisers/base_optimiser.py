from abc import ABC, abstractmethod

from optuna import Trial, Study, create_study


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

    def run(self, n_trials: int = 50) -> None:
        """
        Run the hyperparameter optimization.
        :param n_trials: The number of trials to perform. Default: 50.
        :return: None.
        """

        study: Study = create_study(direction="maximize")
        try:
            study.optimize(self._objective, n_trials=n_trials)
        except NotImplementedError:
            print("The objective function has not been implemented.")
            raise
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping the optimization.")
