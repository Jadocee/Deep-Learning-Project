import logging
import os.path
from abc import ABC, abstractmethod
from os.path import join
from pathlib import Path
from sys import stdout
from typing import List, Optional

from matplotlib.axes import Axes
from optuna import Trial, Study, create_study, visualization
from optuna import logging as optuna_logging
from optuna.pruners import MedianPruner, NopPruner
from optuna.trial import FrozenTrial, TrialState
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history
from pandas import DataFrame

from utils.definitions import STUDIES_DIR


class BaseOptimiser(ABC):
    """
    Base class for a hyperparameter optimiser using Optuna. This class should be subclassed and the _objective method
    should be implemented for the specific problem at hand.
    """
    _device: str

    def __init__(self, device: str = "cpu") -> None:
        """
        Initializes the BaseOptimiser class.

        Args:
            device (str, optional): The device to use for computations. Defaults to "cpu".
        """
        super().__init__()
        self._device = device

    @abstractmethod
    def _objective(self, trial: Trial) -> float:
        """
        Abstract objective function for Optuna hyperparameter optimisation. This method should be implemented in
        subclasses to define the specific problem to optimise.

        Args:
            trial (Trial): Optuna's trial object containing hyperparameters.

        Returns:
            float: Objective function value for the given trial.
        """
        raise NotImplementedError

    def run(self, study_name: Optional[str] = None, n_trials: int = 100, prune: bool = True,
            n_startup_trials: Optional[int] = None, n_warmup_steps: int = 10,
            visualisations: Optional[List[str]] = None) -> None:
        """
        Runs the hyperparameter optimisation process.

        Args:
            visualisations (List[str], optional): The list of Optuna visualisations to generate. If not specified,
                no visualisations will be generated. Defaults to None.
            n_warmup_steps (int, optional): The number of steps to be performed during each trial, before pruning is
                enabled, where a step can be an iteration of training or a forward pass of the model, such as an epoch.
                The warmup steps are usually used to allow the model to converge to a reasonable solution before
                pruning is enabled. Defaults to 10.
            n_startup_trials (int, optional): The number of trials to be performed before pruning is enabled. If not
                specified, defaults to 10% of the total number of trials. Defaults to None.
            study_name (str, optional): The name used to identify the study. If not specified, uses the default study
                name provided by Optuna. Defaults to None.
            n_trials (int, optional): The number of trials to perform. A higher number of trials will result in a more
                accurate optimisation, but will take longer. Defaults to 100.
            prune (bool, optional): If True, enables the pruning feature of Optuna. Defaults to True.

        Returns:
            None.
        """
        optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(stdout))

        # Create the study
        study: Study = create_study(
            direction="maximize",
            study_name=study_name,
            pruner=MedianPruner(n_warmup_steps=n_warmup_steps,
                                n_startup_trials=n_startup_trials if n_startup_trials
                                else n_trials * 0.1) if prune else NopPruner()
        )
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

            output_dir: str = join(STUDIES_DIR, study.study_name)
            if not os.path.exists(output_dir):
                output_path: Path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            # Create visualisations and save them to a file
            if visualisations and len(visualisations) > 0:
                for vis in visualisations:
                    if vis == "slice":
                        ax: Axes = visualization.plot_slice(study)
                        ax.set_title("Slice Plot")
                        ax.figure.savefig(join(output_dir, "slice.png"))
                        print(f"Saved slice plot to {output_dir}")
                    elif vis == "contour":
                        ax: Axes = visualization.plot_contour(study)
                        ax.set_title("Contour Plot")
                        ax.figure.savefig(join(output_dir, "contour.png"))
                        print(f"Saved contour plot to {output_dir}")
                    elif vis == "parallel_coordinate":
                        ax: Axes = visualization.plot_parallel_coordinate(study)
                        ax.set_title("Parallel Coordinate Plot")
                        ax.figure.savefig(join(output_dir, "parallel_coordinate.png"))
                        print(f"Saved parallel coordinate plot to {output_dir}")
                    elif vis == "param_importances":
                        ax: Axes = plot_param_importances(study)
                        ax.set_title("Parameter Importances")
                        ax.figure.savefig(join(output_dir, "param_importances.png"))
                        print(f"Saved parameter importances plot to {output_dir}")
                    elif vis == "optimisation_history":
                        ax: Axes = plot_optimization_history(study)
                        ax.set_title("Optimisation History")
                        ax.figure.savefig(join(output_dir, "optimisation_history.png"))
                        print(f"Saved optimisation history plot to {output_dir}")
                    else:
                        print(f"Invalid visualisation: {vis}")

            # Save the results to a CSV file
            df: DataFrame = study.trials_dataframe()
            tria
            out_csv: str = join(output_dir, "results.csv")
            df.to_csv(out_csv, index=False)
            print(f"Saved results to {out_csv}")

        except NotImplementedError:
            print("The objective function has not been implemented.")
            raise
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping the optimization.")
