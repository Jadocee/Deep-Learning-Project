from os import makedirs
from os.path import exists, join
from typing import List, Dict

from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from seaborn import heatmap


class ResultsUtils:

    @staticmethod
    def __save_plot(file_name: str, save_path: str):
        if not exists(save_path):
            makedirs(save_path, exist_ok=True)
        plt.savefig(join(save_path, file_name))
        plt.close()

    @staticmethod
    def plot_loss_and_accuracy_curves(training_losses: List[float], validation_losses: List[float],
                                      training_accuracies: List[float], validation_accuracies: List[float],
                                      save_path: str) -> None:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
        ax1.plot(training_losses, color='b', label='train')
        ax1.plot(validation_losses, color='r', label='valid')
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax2.plot(training_accuracies, color='b', label='train')
        ax2.plot(validation_accuracies, color='r', label='valid')
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ResultsUtils.__save_plot("loss_and_accuracy_curves.png", save_path)

    @staticmethod
    def plot_confusion_matrix(cm: ndarray, save_path: str) -> None:
        heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        ResultsUtils.__save_plot("confusion_matrix.png", save_path)

    @staticmethod
    def record_performance_scores(scores: Dict[str, float], save_path: str) -> None:
        DataFrame(scores, index=[0]) \
            .rename(columns={"accuracy": "Accuracy", "precision": "Precision",
                             "recall": "Recall", "f1": "F1", "roc_auc": "AUC", "log_loss": "Log Loss"}) \
            .applymap(lambda x: f"{x * 100:.2f}%" if x != "AUC" and x != "Log Loss" else f"{x:.4f}") \
            .transpose() \
            .rename(columns={0: "Value"}) \
            .rename_axis("Score") \
            .to_csv(join(save_path, "performance_scores.csv"), index=True, header=True)
