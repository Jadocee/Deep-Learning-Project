from os import makedirs
from os.path import exists, join
from typing import List

from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import read_csv, DataFrame
from seaborn import heatmap

from utils.definitions import STUDIES_DIR


class ResultsUtils:

    @staticmethod
    def format_study_table(study_name: str):
        path: str = f"{STUDIES_DIR}/{study_name}.csv"
        df: DataFrame = read_csv(path, header=0, index_col=None, delimiter=",")
        # Change all the columns with decimal values to 3 decimal places
        # for col in df.columns:
        #     if col.startswith("params_"):
        #         if df[col].dtype == "float":
        #             # Change to scientific notation only if the value will have more than e
        #             # df[col] = df[col].apply(lambda x: f"{x:.2e}")
        #         # df.rename(columns={col: col[7:]}, inplace=True)
        #     elif col == "Value":
        #         df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%")

        # Remove all columns starting with params_
        df.drop(columns=[col for col in df.columns if col.startswith("params_")], inplace=True)
        df.drop(columns=["datetime_start", "datetime_complete", "duration", "state"], inplace=True)
        df.rename(columns={"value": "Accuracy", "number": "Trial"}, inplace=True)
        df["Accuracy"] = df["Accuracy"].apply(lambda x: f"{(x * 100):.2f}%")
        df["Trial"] = df["Trial"].apply(lambda x: f"{(x + 1):02d}")

        print(df.to_string(header=True, index=False))

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
