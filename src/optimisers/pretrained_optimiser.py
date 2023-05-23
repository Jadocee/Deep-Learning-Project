from datetime import datetime
from os.path import join
from typing import Tuple, Optional, List

import evaluate
import torch
from optuna.trial import FixedTrial

from datasets import DatasetDict
from evaluate import EvaluationModule
from numpy import ndarray, mean, concatenate
from optuna import Trial, TrialPruned
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, roc_auc_score, \
    log_loss
from torch import no_grad, argmax, Tensor, FloatTensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, \
    RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from optimisers.base_optimiser import BaseOptimiser
from utils.dataset_loader import DatasetLoader
from utils.definitions import STUDIES_DIR
from utils.hyperparam_utils import HyperParamUtils
from utils.results_utils import ResultsUtils


class PretrainedOptimiser(BaseOptimiser):
    __pretrained_model_name: str

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(device=device)
        self.__pretrained_model_name = model_name

    def _prepare_data(self, batch_size: int, max_tokens: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset_dict: DatasetDict = DatasetLoader.get_tweet_topic_single_dataset()
        tokenizer = AutoTokenizer.from_pretrained(self.__pretrained_model_name)
        dataset_dict = dataset_dict.map(
            lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=max_tokens),
            batched=True,
            remove_columns=["text"]
        )
        dataset_dict = dataset_dict.rename_column("label", "labels")
        dataset_dict.set_format(type="torch", columns=["input_ids", "labels"])
        print(dataset_dict.column_names.__str__())
        train_dataloader = DataLoader(dataset_dict["train"], batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset_dict["validation"], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset_dict["test"], batch_size=batch_size)
        return train_dataloader, valid_dataloader, test_dataloader

    def test_model(self, trial: FixedTrial) -> None:
        optimiser_name: str = trial.suggest_categorical("optimiser", ["AdamW", "Adam"])
        lr: float = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        batch_size: int = trial.suggest_categorical("batch_size", [8])
        epochs: int = trial.suggest_categorical("epochs", [3, 5, 10])
        scheduler: str = trial.suggest_categorical("scheduler", ['linear', 'cosine', 'cosine_with_restarts',
                                                                 'polynomial', 'constant', 'constant_with_warmup',
                                                                 'inverse_sqrt', 'reduce_lr_on_plateau'])
        train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                 max_tokens=200)
        model: RobertaForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            self.__pretrained_model_name, num_labels=6)
        optimiser: Optimizer = HyperParamUtils.define_optimiser(optimiser_name=optimiser_name,
                                                                model_params=model.parameters(), learning_rate=lr)
        num_training_steps: int = len(test_dataloader)
        lr_scheduler = get_scheduler(name=scheduler, optimizer=optimiser, num_training_steps=num_training_steps,
                                     num_warmup_steps=1)
        model.to(self._device)

        progress_bar: tqdm = tqdm(range(num_training_steps), desc=f"Training {self.__pretrained_model_name}",
                                  unit="epoch")
        model.train()
        for epoch in range(epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                outputs: SequenceClassifierOutput = model(**batch)
                loss: Optional[FloatTensor] = outputs.loss
                loss.backward()
                optimiser.step()
                lr_scheduler.step(epoch=epoch)
                optimiser.zero_grad()
                progress_bar.update(1)
                predictions: Tensor = argmax(outputs.logits, dim=-1)

        model.eval()
        progress_bar.set_description_str(f"Testing {self.__pretrained_model_name}")
        preds: ndarray = ndarray(shape=(0,), dtype=int)
        targets: ndarray = ndarray(shape=(0,), dtype=int)
        for batch in test_dataloader:
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with no_grad():
                outputs: SequenceClassifierOutput = model(**batch)
            logits: FloatTensor = outputs.logits
            predictions: Tensor = argmax(logits, dim=-1)
            progress_bar.update(1)
            preds = concatenate((preds, predictions.cpu().numpy()))
            targets = concatenate((targets, batch["labels"].cpu().numpy()))

        save_path: str = join(STUDIES_DIR,
                              f"Test_{self.__pretrained_model_name}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")

        ResultsUtils.record_performance_scores(
            scores={
                "accuracy": accuracy_score(y_true=targets, y_pred=preds),
                "precision": precision_score(y_true=targets, y_pred=preds, average="macro"),
                "f1": f1_score(y_true=targets, y_pred=preds, average="macro"),
                "recall": recall_score(y_true=targets, y_pred=preds, average="macro"),
                "roc_auc": roc_auc_score(y_true=targets, y_score=preds),
                "log_loss": log_loss(y_true=targets, y_pred=preds),
            },
            save_path=save_path
        )

        ResultsUtils.plot_confusion_matrix(
            cm=confusion_matrix(y_true=targets, y_pred=preds),
            save_path=save_path
        )

    def _objective(self, trial: Trial) -> float:
        batch_size: int = trial.suggest_categorical("batch_size", [8])
        epochs: int = trial.suggest_categorical("epochs", [10])
        lr: float = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        optimiser_name: str = trial.suggest_categorical("optimiser", ["Adam"])
        scheduler: str = trial.suggest_categorical("scheduler", ["cosine"])

        train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                 max_tokens=200)

        model: RobertaForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            self.__pretrained_model_name, num_labels=6)
        optimiser: Optimizer = HyperParamUtils.define_optimiser(optimiser_name=optimiser_name,
                                                                model_params=model.parameters(), learning_rate=lr)
        num_training_steps: int = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(name=scheduler, optimizer=optimiser, num_training_steps=num_training_steps,
                                     num_warmup_steps=1)
        model.to(self._device)

        progress_bar: tqdm = tqdm(range(num_training_steps), desc=f"Training {self.__pretrained_model_name}",
                                  unit="epoch")
        avg_train_losses: List[float] = list()
        avg_valid_losses: List[float] = list()
        avg_train_accuracies: List[float] = list()
        avg_valid_accuracies: List[float] = list()
        training_metric: EvaluationModule = evaluate.load("accuracy")
        validation_metric = evaluate.load("accuracy")
        for epoch in range(epochs):
            train_losses: List[ndarray] = list()
            valid_losses: List[ndarray] = list()
            train_accuracies: List[ndarray] = list()
            valid_accuracies: List[ndarray] = list()
            model.train()
            for batch in train_dataloader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                outputs: SequenceClassifierOutput = model(**batch)
                loss: Optional[FloatTensor] = outputs.loss
                loss.backward()
                optimiser.step()
                lr_scheduler.step(epoch=epoch)
                optimiser.zero_grad()
                progress_bar.update(1)
                predictions: Tensor = argmax(outputs.logits, dim=-1)
                training_metric.add_batch(predictions=predictions, references=batch["labels"])
                accuracy: Tensor = torch.sum(predictions == batch["labels"]).float() / predictions.shape[0]
                train_losses.append(loss.detach().cpu().numpy())
                train_accuracies.append(accuracy.detach().cpu().numpy())
            avg_train_losses.append(mean(train_losses))
            avg_train_accuracies.append(mean(train_accuracies))
            model.eval()
            for batch in valid_dataloader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                with no_grad():
                    outputs: SequenceClassifierOutput = model(**batch)
                logits: FloatTensor = outputs.logits
                predictions: Tensor = argmax(logits, dim=-1)
                loss: Optional[FloatTensor] = outputs.loss
                valid_losses.append(loss.detach().cpu().numpy())
                accuracy: Tensor = torch.sum(predictions == batch["labels"]).float() / predictions.shape[0]
                valid_accuracies.append(accuracy.detach().cpu().numpy())
                validation_metric.add_batch(predictions=predictions, references=batch["labels"])
            avg_valid_losses.append(mean(valid_losses))
            avg_valid_accuracies.append(mean(valid_accuracies))
            progress_bar.set_postfix({"train_acc": f"{avg_train_accuracies[-1] * 100:.2f}%",
                                      "valid_acc": f"{avg_valid_accuracies[-1] * 100:.2f}%"})
            trial.report(avg_valid_accuracies[-1], epoch)
            if trial.should_prune():
                raise TrialPruned()

        save_path: str = join(STUDIES_DIR, trial.study.study_name,
                              f"trial_{trial.number}_{self.__pretrained_model_name}")
        trial.set_user_attr(key="save_path", value=save_path)
        ResultsUtils.plot_loss_and_accuracy_curves(
            training_losses=avg_train_losses,
            validation_losses=avg_valid_losses,
            training_accuracies=avg_train_accuracies,
            validation_accuracies=avg_valid_accuracies,
            save_path=save_path
        )
        return validation_metric.compute()["accuracy"]
