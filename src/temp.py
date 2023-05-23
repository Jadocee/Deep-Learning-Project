from os.path import join
from typing import Any, Dict, List

import pandas as pd
from optuna.trial import FixedTrial
from pandas import DataFrame

from utils.definitions import STUDIES_DIR


def foo(self, study_name):
    df: DataFrame = pd.read_csv(join(STUDIES_DIR, study_name, "results.csv")) \
        .astype({'params_learning_rate': float, 'params_max_tokens': int, 'params_epochs': int,
                 'params_batch_size': int, 'params_optimiser': str, "params_dropout": float,
                 "params_bidirectional": bool, "params_hidden_size": int, "params_n_layers": int}) \
        .sort_values(by=["value"], ascending=False) \
        .head(3)

    for i, row in df.iterrows():
        params_dict: Dict[str, Any] = dict()
        if row['param_lr_scheduler'] is not None:
            if row['param_lr_scheduler'] == "MultiStepLR":
                n_milestones: int = row['params_n_milestones'].astype(int)
                milestones: List[int] = [row[f"params_milestone_{i}"].astype(int) for i in range(n_milestones)]
                gamma: float = row['params_gamma'].astype(float)
                params_dict.update({"milestones": milestones, "gamma": gamma, "lr_scheduler": "MultiStepLR"})
            elif row['param_lr_scheduler'] == "ReduceLROnPlateau":
                mode: str = row['params_mode'].astype(str)
                factor: float = row['params_factor'].astype(float)
                patience: int = row['params_patience'].astype(int)
                threshold: float = row['params_threshold'].astype(float)
                threshold_mode: str = row['params_threshold_mode'].astype(str)
                cooldown: int = row['params_cooldown'].astype(int)
                min_lr: float = row['params_min_lr'].astype(float)
                eps: float = row['params_eps'].astype(float)
                params_dict.update({"mode": mode, "factor": factor, "patience": patience, "threshold": threshold,
                                    "threshold_mode": threshold_mode, "cooldown": cooldown, "min_lr": min_lr,
                                    "eps": eps, "lr_scheduler": "ReduceLROnPlateau"})
            elif row['param_lr_scheduler'] == "CosineAnnealingLR":
                T_max: int = row['params_T_max'].astype(int)
                eta_min: float = row['params_eta_min'].astype(float)
                params_dict.update({"T_max": T_max, "eta_min": eta_min, "lr_scheduler": "CosineAnnealingLR"})
            elif row['param_lr_scheduler'] == "StepLR":
                step_size: int = row['params_step_size'].astype(int)
                gamma: float = row['params_gamma'].astype(float)
                params_dict.update({"step_size": step_size, "gamma": gamma, "lr_scheduler": "StepLR"})
            elif row['param_lr_scheduler'] == "ExponentialLR":
                gamma: float = row['params_gamma'].astype(float)
                params_dict.update({"gamma": gamma, "lr_scheduler": "ExponentialLR"})
            else:
                raise ValueError(f"Unknown lr_scheduler: {row['param_lr_scheduler']}")
        else:
            params_dict.update({"lr_scheduler": None})

        params_dict.update({"learning_rate": row['params_learning_rate'],
                            "max_tokens": row['params_max_tokens'],
                            "epochs": row['params_epochs'],
                            "batch_size": row['params_batch_size'],
                            "optimiser": row['params_optimiser'],
                            "bidirectional": row['params_bidirectional'],
                            "hidden_size": row['params_hidden_size'],
                            "n_layers": row['params_n_layers']})

        if "params_dropout" in row.keys():
            params_dict.update({"dropout": row['params_dropout']})
        trial: FixedTrial = FixedTrial(params_dict)
