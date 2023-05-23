from typing import List, Iterable, Any, Union

from torch import optim
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, \
    ReduceLROnPlateau


class HyperParamUtils:

    def __init__(self):
        raise Exception("This class is not meant to be instantiated.")

    @staticmethod
    def define_optimiser(optimiser_name: str, model_params: Iterable[Parameter], learning_rate: float) -> Optimizer:
        """
        Get an optimiser from a name.

        Args:
            model_params (Iterable[Parameter]): The parameters of the model.
            optimiser_name (str): The name of the optimiser.
            learning_rate (float): The learning rate to use.

        Raises:
            ValueError: If the optimiser name is not recognised.

        Returns:
            Optimizer: The optimiser.
        """
        if optimiser_name not in dir(optim):
            raise ValueError(f"Unknown optimiser: {optimiser_name}")
        return getattr(optim, optimiser_name)(model_params, lr=learning_rate)

    @staticmethod
    def define_scheduler(learning_rate_scheduler_name: str, optimiser: Optimizer, **kwargs: Any) \
            -> Union[LRScheduler, ReduceLROnPlateau]:
        """
        Get a learning rate scheduler from a name.

        Notes:
            The kwargs are specific to each learning rate scheduler.

            For `StepLR`, the kwargs are:
                - `step_size` (int): Period of learning rate decay.
                - `gamma` (float): Multiplicative factor of learning rate decay.
            For `MultiStepLR`, the kwargs are:
                - `milestones` (List[int]): List of epoch indices. Must be increasing.
                - `gamma` (float): Multiplicative factor of learning rate decay.
            For `ExponentialLR`, the kwargs are:
                - `gamma` (float): Multiplicative factor of learning rate decay.
            For `CosineAnnealingLR`, the kwargs are:
                - `T_max` (int): Maximum number of iterations.
                - `eta_min` (float): Minimum learning rate. Default: 0.
            For `ReduceLROnPlateau`, the kwargs are:
                - `threshold` (float): Threshold for measuring the new optimum, to only focus on significant changes.
                - `patience` (int): Number of epochs with no improvement after which learning rate will be reduced.
                - `factor` (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
                - `min_lr` (float): A scalar or a list of scalars. A lower bound on the learning rate of all param groups
                - `threshold_mode` (str): One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in
                  'max' mode or best * ( 1 - threshold ) in 'min' mode. In abs mode, dynamic_threshold = best +
                  threshold in max mode or best - threshold in min mode. Default: 'rel'.


        Args:
            optimiser (Optimizer): The optimiser to use.
            learning_rate_scheduler_name (str): The name of the learning rate scheduler.
            **kwargs (Any): The arguments for the learning rate scheduler. Must be valid for the chosen learning rate
            scheduler.

        Raises:
            ValueError: If the learning rate scheduler name is not recognised or if an unknown kwarg is passed.

        Returns:
            LRScheduler: The learning rate scheduler.
        """
        if learning_rate_scheduler_name not in ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR",
                                                "ReduceLROnPlateau"]:
            raise ValueError(f"Unknown learning rate scheduler: {learning_rate_scheduler_name}")

        if learning_rate_scheduler_name == "StepLR":
            accepted_kwargs: List[str] = ["step_size", "gamma", "last_epoch", "verbose"]
            for kwarg in kwargs:
                if kwarg not in accepted_kwargs:
                    raise ValueError(f"Unknown argument for StepLR: {kwarg}")
            return StepLR(optimizer=optimiser, **kwargs)
        elif learning_rate_scheduler_name == "MultiStepLR":
            accepted_kwargs: List[str] = ["milestones", "gamma", "last_epoch", "verbose"]
            for kwarg in kwargs:
                if kwarg not in accepted_kwargs:
                    raise ValueError(f"Unknown argument for MultiStepLR: {kwarg}")
            return MultiStepLR(optimizer=optimiser, **kwargs)
        elif learning_rate_scheduler_name == "ExponentialLR":
            accepted_kwargs: List[str] = ["gamma", "last_epoch", "verbose"]
            for kwarg in kwargs:
                if kwarg not in accepted_kwargs:
                    raise ValueError(f"Unknown argument for ExponentialLR: {kwarg}")
            return ExponentialLR(optimizer=optimiser, **kwargs)
        elif learning_rate_scheduler_name == "CosineAnnealingLR":
            return CosineAnnealingLR(optimizer=optimiser, **kwargs)
        elif learning_rate_scheduler_name == "ReduceLROnPlateau":
            accepted_kwargs: List[str] = ["mode", "factor", "patience", "verbose", "threshold", "threshold_mode",
                                          "cooldown", "min_lr", "eps"]
            for kwarg in kwargs:
                if kwarg not in accepted_kwargs:
                    raise ValueError(f"Unknown argument for ReduceLRonPlateau: {kwarg}")
            return ReduceLROnPlateau(optimizer=optimiser, **kwargs)
