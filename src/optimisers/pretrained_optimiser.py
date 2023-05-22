from typing import Tuple, Optional, Any, Dict

from datasets import DatasetDict
from evaluate import EvaluationModule
from optuna import Trial
from torch import no_grad, argmax
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import evaluate

from optimisers.base_optimiser import BaseOptimiser
from utils.dataset_loader import DatasetLoader
from utils.hyperparam_utils import HyperParamUtils


class PretrainedOptimiser(BaseOptimiser):
    __pretrained_model_name: str

    def __init__(self, model_name: str, device: str = "cpu", ):
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

    def _objective(self, trial: Trial) -> float:
        torch.cuda.empty_cache()
        batch_size: int = trial.suggest_categorical("batch_size", [8])
        epochs: int = trial.suggest_categorical("epochs", [3, 5, 10])
        lr: float = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        optimiser_name: str = trial.suggest_categorical("optimiser", ["AdamW", "Adam"])
        scheduler: str = trial.suggest_categorical("scheduler", ['linear', 'cosine', 'cosine_with_restarts',
                                                                 'polynomial', 'constant', 'constant_with_warmup',
                                                                 'inverse_sqrt', 'reduce_lr_on_plateau'])

        train_dataloader, valid_dataloader, test_dataloader = self._prepare_data(batch_size=batch_size,
                                                                                 max_tokens=200)

        model = AutoModelForSequenceClassification.from_pretrained(self.__pretrained_model_name, num_labels=6)
        optimiser: Optimizer = HyperParamUtils.define_optimiser(optimiser_name=optimiser_name,
                                                                model_params=model.parameters(), learning_rate=lr)
        num_training_steps = epochs * len(train_dataloader)
        # num_warmpup_steps = epochs * len(train_dataloader) // 10
        lr_scheduler = get_scheduler(name=scheduler, optimizer=optimiser, num_training_steps=num_training_steps,
                                     num_warmup_steps=0)
        model.to(self._device)

        progress_bar = tqdm(range(num_training_steps))
        training_metric: EvaluationModule = evaluate.load("accuracy")
        model.train()
        for epoch in range(epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimiser.step()
                lr_scheduler.step(epoch)
                optimiser.zero_grad()
                progress_bar.update(1)
                predictions = argmax(outputs.logits, dim=-1)
                training_metric.add_batch(predictions=predictions, references=batch["labels"])


        metric = evaluate.load("accuracy")
        model.eval()
        for batch in valid_dataloader:
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["label"])
            print(metric.compute())
        return metric.compute()
