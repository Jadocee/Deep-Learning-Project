import os.path
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger, StreamHandler, INFO
from os.path import join
from pathlib import Path
from sys import stdout
from typing import List, Optional, Dict, Any, Tuple, Final, Set

import pandas as pd
from matplotlib.axes import Axes
from optuna import Trial, Study, create_study
from optuna import logging as optuna_logging
from optuna.pruners import MedianPruner, NopPruner, BasePruner, HyperbandPruner
from optuna.study import StudyDirection
from optuna.trial import FixedTrial
from optuna.trial import FrozenTrial, TrialState
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainers.base_trainer import BaseTrainer
from utils.definitions import STUDIES_DIR


class BaseOptimiser(ABC):
    """
    Base class for a hyperparameter optimiser using Optuna.

    This class should be subclassed and the _objective method should be implemented for the specific problem at hand.

    Attributes:
        _device (str): The device to use for computations.
        _logger (Logger): The logger for the Optuna study.
        OPTUNA_VISUALISATION_NAMES (Set[str]): The names of the visualisations that can be used for the Optuna study;
            see link in references for more information.
    References:
        - https://optuna.readthedocs.io/en/stable/index.html
        - https://optuna.readthedocs.io/en/stable/reference/visualization/index.html
    """

    OPTUNA_VISUALISATION_NAMES: Final[List[str]] = ["slice_plot", "contour_plot", "param_importances_plot",
                                                    "optimization_history_plot", "parallel_coordinate_plot"]

    _device: str
    _logger: Final[Logger]
    _progress_bar: tqdm

    def __init__(self, device: str = "cpu") -> None:
        """
        Constructor for the `BaseOptimiser` class.

        Initializes the `BaseOptimiser` class by setting the device to use for computations.

        Args:
            device (str, optional): The device to use for computations. Defaults to "cpu".

        Returns:
            None
        """

        super().__init__()
        self._device = device
        self._logger = optuna_logging.get_logger("optuna")
        self._logger.setLevel(INFO)
        self._logger.addHandler(StreamHandler(stdout))

    @staticmethod
    def __create_pruner(pruning_algorithm: str, kwargs: Optional[Dict[str, Any]] = None,
                        n_trials: Optional[int] = None) -> BasePruner:
        """
        Creates a pruner for the Optuna study.

        Args:
            pruning_algorithm (str): The pruning algorithm to use.
            kwargs (Optional[Dict[str, Any]], optional): The hyperparameters for the pruning algorithm. Defaults to None.
            n_trials (Optional[int], optional): The number of trials to run. Defaults to None.

        Returns:
            BasePruner: The pruner for the Optuna study.
        """

        if pruning_algorithm == "nop":
            return NopPruner()

        elif pruning_algorithm == "median":
            if kwargs is None:
                return MedianPruner(n_warmup_steps=5, interval_steps=1, n_startup_trials=int(n_trials * 0.1))

            if "n_startup_trials" not in kwargs:
                kwargs["n_startup_trials"] = int(n_trials * 0.1)
            return MedianPruner(**kwargs)

        elif pruning_algorithm == "hyperband":
            if kwargs is None:
                return HyperbandPruner()

            return HyperbandPruner(**kwargs)

        else:
            raise ValueError(f"Unknown pruning algorithm: {pruning_algorithm}")

    @staticmethod
    def _define_scheduler_hyperparams(trial: Trial) -> Optional[Dict[str, Any]]:
        """
        Defines the hyperparameters for the scheduler.

        An optional method for experimenting with various learning rate schedulers and respective hyperparameters.

        The format of the output is as follows:

        `{"scheduler": <scheduler_name>, "kwargs": { <scheduler_hyperparam_1>: <value_1>,
        <scheduler_hyperparam_2>: <value_2>, ... }}`

        Args:
            trial (Trial): The trial object of the relevant Optuna study.

        Returns:
            Optional[Dict[str, Any]]: The hyperparameters for the scheduler.

        """

        lr_scheduler_name: Optional[str] = trial.suggest_categorical("lr_scheduler", [
            "StepLR", "ExponentialLR", "MultiStepLR", None, "ReduceLROnPlateau", "CosineAnnealingLR"])
        if lr_scheduler_name is None:
            return None

        kwargs: Dict[str, Any] = dict()
        if lr_scheduler_name == "StepLR":
            step_size: int = trial.suggest_int("step_size", 1, trial.params["epochs"])
            gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
            kwargs.update({"step_size": step_size, "gamma": gamma})
        elif lr_scheduler_name == "ExponentialLR":
            gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
            kwargs.update({"gamma": gamma})
        elif lr_scheduler_name == "MultiStepLR":
            n_milestones: int = trial.suggest_int("n_milestones", 1, 5)
            milestones: List[int] = list()
            for i in range(n_milestones):
                milestones.append(trial.suggest_int(f"milestone_{i}", 1, trial.params["epochs"]))
            gamma: float = trial.suggest_float("gamma", 0.1, 0.9)
            kwargs.update({"milestones": milestones, "gamma": gamma})
        elif lr_scheduler_name == "ReduceLROnPlateau":
            factor: float = trial.suggest_float("factor", 0.1, 0.9)
            patience: int = trial.suggest_int("patience", 1, 10)
            threshold: float = trial.suggest_float("threshold", 1e-4, 1e-1)
            threshold_mode: str = trial.suggest_categorical("threshold_mode", ["rel", "abs"])
            cooldown: int = trial.suggest_int("cooldown", 1, 10)
            min_lr: float = trial.suggest_float("min_lr", 1e-6, 1e-1, log=True)
            kwargs.update({"factor": factor, "patience": patience, "threshold": threshold,
                           "threshold_mode": threshold_mode, "cooldown": cooldown, "min_lr": min_lr})
        elif lr_scheduler_name == "CosineAnnealingLR":
            T_max: int = trial.suggest_int("T_max", 1, trial.params["epochs"])
            eta_min: float = trial.suggest_float("eta_min", 1e-4, 1e-1)
            kwargs.update({"T_max": T_max, "eta_min": eta_min})

        return {
            "name": lr_scheduler_name,
            "kwargs": kwargs
        }

    @abstractmethod
    def _prepare_data(self, batch_size: int, max_tokens: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Abstract method for preparing the data for the hyperparameter optimisation process.

        This method should be implemented in subclasses to prepare the training, validation, and test data for the
        hyperparameter optimisation process.

        Any unassigned attributes of the implementing class should be assigned in this method. For example, if the
        implementing class has a `vocab` attribute that is required for and is created during the data preparation
        process, then the `vocab` attribute should be assigned in this method.

        Notes:
            - This method is a required step in the hyperparameter optimisation process; it is called before the
                hyperparameter optimisation process begins.
            - The goal of this method is to reduce the number of times the data preparation process is executed. This
                is achieved by reusing the same instance of the trainer for each trial of the hyperparameter
                optimisation process.

        Returns:
            BaseTrainer: The trainer implementation to use for the hyperparameter optimisation process.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """

        raise NotImplementedError

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

    def run(self, n_trials: int = 100) -> None:
        """
        Runs the hyperparameter optimisation process.

        This method is responsible for running the hyperparameter optimisation process. It is responsible for creating
        the Optuna study, and for generating the visualisations. It also calls the `_objective` method to define the
        specific problem to optimise.

        The Hyperband pruner is used to prune unpromising trials. The pruner is used to stop unpromising trials at
        early stages of the training process, which reduces the number of trials that are executed. This is achieved
        by using the `reduction_factor` argument of the Hyperband pruner. For example, if `n_trials` is 100, and
        `reduction_factor` is 3, then roughly one third of trials will be retained for the next round, i.e., the first
        round of Hyperband will execute 20 trials, and the second round will execute 7 trials.

        Args:
            n_trials (int, optional): The number of trials to perform. A higher number of trials will result in a more
                accurate optimisation, but will take longer. Defaults to 100.

        Returns:
            None

        References:
            - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html
        """

        study: Study = create_study(
            direction=StudyDirection.MAXIMIZE,
            study_name=f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            pruner=HyperbandPruner(min_resource=3, max_resource="auto", reduction_factor=3)
        )
        try:
            study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True)
            pruned_trials: List[FrozenTrial] = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials: List[FrozenTrial] = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Summary: ")
            print(f"\tNumber of finished trials: {len(study.trials)}")
            print(f"\tNumber of pruned trials: {len(pruned_trials)}")
            print(f"\tNumber of complete trials: {len(complete_trials)}")
            print(f"Best trial:")
            best_trial: FrozenTrial = study.best_trial
            print(f"\tValue: {best_trial.value}")
            print("Params: ")
            for key, value in best_trial.params.items():
                print(f"\t{key}: {value}")

            output_dir: str = join(STUDIES_DIR, study.study_name)
            if not os.path.exists(output_dir):
                output_path: Path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                # Ignore 'ExperimentalWarning' warnings from Optuna
                warnings.simplefilter("ignore")
                ax: Axes = plot_param_importances(study)
                ax.set_title("Parameter Importances")
                ax.figure.savefig(join(output_dir, "param_importances.png"))
                ax.clear()
                ax: Axes = plot_optimization_history(study)
                ax.set_title("Optimisation History")
                ax.figure.savefig(join(output_dir, "optimisation_history.png"))
                ax.clear()
                print(f"Saved visualisations to {output_dir}")

            df: DataFrame = study.trials_dataframe()
            out_csv: str = join(output_dir, "results.csv")
            df.to_csv(out_csv, index=False)
            print(f"Saved results to {out_csv}")

        except NotImplementedError:
            print("The objective function has not been implemented.")
            raise
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping the optimization.")

    def create_fixed_trial(self, study_name):
        # Read the CSV file
        df: DataFrame = pd.read_csv(join(STUDIES_DIR, study_name, "results.csv")) \
            .astype({'params_learning_rate': float, 'params_epochs': int,
                     'params_batch_size': int, 'params_optimiser': str, "params_hidden_size": int,
                     "params_n_layers": int}) \
            .sort_values(by=["value"], ascending=False) \
            .head(3)
        for i, row in df.iterrows():
            params_dict: Dict[str, Any] = dict()
            print(row['params_lr_scheduler'])
            if pd.notnull(row['params_lr_scheduler']):
                if row['params_lr_scheduler'] == "MultiStepLR":
                    n_milestones: int = int(row['params_n_milestones'])
                    milestones: List[int] = [int(row[f"params_milestone_{i}"]) for i in range(n_milestones)]
                    for i in range(n_milestones):
                        params_dict.update(
                            {f"milestone_{i}": milestones[i]}
                        )
                    gamma: float = row['params_gamma']
                    params_dict.update(
                        {"n_milestones": n_milestones, "gamma": gamma, "lr_scheduler": "MultiStepLR"})
                elif row['params_lr_scheduler'] == "ReduceLROnPlateau":
                    factor: float = row['params_factor']
                    patience: int = row['params_patience']
                    threshold: float = row['params_threshold']
                    threshold_mode: str = row['params_threshold_mode']
                    cooldown: int = row['params_cooldown']
                    min_lr: float = row['params_min_lr']
                    params_dict.update({"factor": factor, "patience": patience, "threshold": threshold,
                                        "threshold_mode": threshold_mode, "cooldown": cooldown, "min_lr": min_lr,
                                        "lr_scheduler": "ReduceLROnPlateau"})
                elif row['params_lr_scheduler'] == "CosineAnnealingLR":
                    T_max: int = row['params_T_max']
                    eta_min: float = row['params_eta_min']
                    params_dict.update(
                        {"T_max": T_max, "eta_min": eta_min, "lr_scheduler": "CosineAnnealingLR"})
                elif row['params_lr_scheduler'] == "StepLR":
                    step_size: int = row['params_step_size']
                    gamma: float = row['params_gamma']
                    params_dict.update(
                        {"step_size": step_size, "gamma": gamma, "lr_scheduler": "StepLR"})
                elif row['params_lr_scheduler'] == "ExponentialLR":
                    gamma: float = row['params_gamma']
                    params_dict.update(
                        {"gamma": gamma, "lr_scheduler": "ExponentialLR"})
                else:
                    raise ValueError(
                        f"Unknown lr_scheduler: {row['params_lr_scheduler']}")
            else:
                params_dict.update({"lr_scheduler": None})
            if "params_dropout" in row:
                params_dict.update({"dropout": row['params_dropout']})
            if "params_embedding_dim" in row:
                params_dict.update({"embedding_dim": row['params_embedding_dim']})
            if "params_bidirectional" in row:
                params_dict.update({"bidirectional": row['params_bidirectional']})
            if 'params_max_tokens' in row:
                params_dict.update({"max_tokens": row['params_max_tokens']})
            params_dict.update({"learning_rate": row['params_learning_rate'],
                                "epochs": row['params_epochs'],
                                "batch_size": row['params_batch_size'],
                                "optimiser": row['params_optimiser'],
                                "hidden_size": row['params_hidden_size'],
                                "n_layers": row['params_n_layers']})
            trial: FixedTrial = FixedTrial(params_dict)
            self._evaluate_test(trial, join(STUDIES_DIR, study_name), i)
