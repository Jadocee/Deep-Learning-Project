from os import system, name
from os.path import exists, join
from pathlib import Path
from typing import Dict, List, Final, Any
import torch

from optuna.trial import FixedTrial
from pandas import read_csv, DataFrame, Series
from pandas._testing import loc
from torch.cuda import is_available as has_cuda, empty_cache

from optimisers.bow_classifier_optimiser import BOWClassifierOptimiser
from optimisers.lstm_classifier_optimiser import LSTMClassifierOptimiser
from optimisers.pretrained_optimiser import PretrainedOptimiser
from trainers.cnn_trainer import menu_prompt
from utils.definitions import MODELS_DIR, STUDIES_DIR


class Main:
    """
    Main class which acts as the command-line interface of the application.

    This class is not meant to be instantiated. It provides several static methods for interacting with the application.

    Attributes:
        DEVICE (str): The device to be used for computations. If CUDA is available, it defaults to "cuda", else "cpu".
        MENUS (Dict[str, Dict[int, str]]): A dictionary containing the structure of the menus to be displayed on the
            command-line interface. The keys are the names of the menus, and the values are dictionaries containing the
            menu items. The keys of the menu items are the numbers to be entered by the user to select the menu item,
            and the values are the names of the menu items. The menu items are displayed in the order of their keys. For
            example, the "Main Menu" has 4 menu items, and the "Tweet Classifier Menu" has 2 menu items. The "Main Menu"
            is displayed first, and the user can select one of the 4 menu items. If the user selects the "Tweet
            Classifier Menu", the "Tweet Classifier Menu" is displayed, and the user can select one of the 2 menu items.
            The user can always go back to the "Main Menu" by entering 0. The user can exit the application by entering
            -1.
    """

    DEVICE: Final[str] = "cuda" if has_cuda() else "cpu"

    MENUS: Final[Dict[str, List[str]]] = {
        "Main Menu": [
            "Image Classifier",
            "Tweet Classifier",
            "Check CUDA Availability",
            "Clear CUDA Cache",
        ],
        "Image Classifier Menu": [
            "TODO"
        ],
        "Tweet Classifier Menu": [
            "LSTM Trainer",
            "BOW Trainer",
            "Fine-Tune Pre-Trained Model",
        ],
        "LSTM Trainer Menu": [
            "Optimise Hyperparameters",
            "Evaluate Top 3 Models on Test Set"
        ],
        "BOW Trainer Menu": [
            "Run Bag Of Words", 
            "Validate Top 10 Models",
            "Evaluate Top 3 Models on Test Set",
            "General Analysis"
        ],
        "Fine-Tune Pre-Trained Model Menu": [
            "Fine-tune cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all",
            "Test cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all",
        ]
    }

    def __init__(self) -> None:
        """
        The constructor for Main class. Raises an exception since the class should not be instantiated.

        Raises:
            Exception: If an attempt is made to instantiate the Main class.
        """
        raise Exception("This class is not meant to be instantiated")

    @staticmethod
    def __clear_cuda_cache() -> None:
        """
        Clears the CUDA cache.

        This method takes no arguments and returns nothing. It clears the CUDA cache by calling the clear() method of
        the torch.cuda module.
        """
        print("Clearing CUDA cache...")
        empty_cache()
        print("Cleared CUDA cache")

    @staticmethod
    def __display_menu(menu_name: str) -> None:
        """
        Displays a specified menu on the command-line interface.

        Args:
            menu_name (str): The name of the menu to be displayed.
        Raises:
            Exception: If the provided menu_name does not exist in the MENUS dictionary.
        """
        if menu_name not in Main.MENUS.keys():
            raise Exception(f"Unknown menu: {menu_name}")

        menu: List[str] = Main.MENUS[menu_name]
        print(f"--- {menu_name} ---")
        for key, value in enumerate(menu, start=1):
            print(f"{key}. {value}")
        if menu_name != "Main Menu":
            print("0. Main Menu")
        print("-1. Exit", end="\n\n")

    @staticmethod
    def __switch_menu():
        """
        Handles user interaction with the command-line interface.

        The method takes no arguments and returns nothing. It enters into an infinite loop where it displays the menu
        and waits for the user to enter a choice. The loop breaks only when the user chooses to exit.
        """
        current_menu = "Main Menu"

        while True:
            Main.__display_menu(current_menu)

            try:
                choice: int = int(input("Enter choice: "))
            except ValueError:
                print("Invalid choice. Try again.")
                system("cls" if name == "nt" else "clear")
                continue

            if choice == -1:
                print("Exiting...")
                break

            if current_menu == "Main Menu":
                if choice == 1:
                    current_menu = "Image Classifier Menu"
                elif choice == 2:
                    current_menu = "Tweet Classifier Menu"
                elif choice == 3:
                    print(f"CUDA is {'available' if has_cuda() else 'not available'}")
                    input("Press any key to continue...")
                    system("cls" if name == "nt" else "clear")
                elif choice == 4:
                    Main.__clear_cuda_cache()
                    input("Press any key to continue...")
                    system("cls" if name == "nt" else "clear")
                elif choice == 5:
                    current_menu = "Fine-Tune Pre-Trained Model"
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
            elif current_menu == "Image Classifier Menu":
                if choice == 0:
                    current_menu = "Main Menu"
                elif choice == 1:
                    system("cls" if name == "nt" else "clear")
                    menu_prompt("resnet18", 50, 50, 0.001)
                    # TODO modify to ask users for hyper params
                else:
                    print("Invalid choice. Try again.")
            elif current_menu == "Tweet Classifier Menu":
                if choice == 0:
                    current_menu = "Main Menu"
                elif choice == 1:
                    current_menu = "LSTM Trainer Menu"
                elif choice == 2:
                    current_menu = "BOW Trainer Menu"
                elif choice == 3:
                    current_menu = "Fine-Tune Pre-Trained Model Menu"
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
            elif current_menu == "LSTM Trainer Menu":
                if choice == 0:
                    current_menu = "Main Menu"
                elif choice == 1:
                    system("cls" if name == "nt" else "clear")
                    optimiser: LSTMClassifierOptimiser = LSTMClassifierOptimiser(device=Main.DEVICE)
                    optimiser.run(n_trials=120)
                elif choice == 2:  
                    optimiser: LSTMClassifierOptimiser = LSTMClassifierOptimiser(device=Main.DEVICE)
                    study_name: str = input(
                        "Please enter study name to evaluate Test Set:"
                    )
                    optimiser.create_fixed_trial(study_name)
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
            elif current_menu == "BOW Trainer Menu":
                cuda = "cuda" if torch.cuda.is_available() else "cpu"
                optimiser: BOWClassifierOptimiser = BOWClassifierOptimiser(device=cuda)
                if choice == 0:
                    current_menu = "Main Menu"
                elif choice == 1:
                    optimiser.run(n_trials=120)
                elif choice == 2:  
                    study_name: str = input(
                        "Please enter study to validate:"
                    )
                    optimiser.validate(study_name)
                elif choice == 3:
                    study_name: str = input(
                        "Please enter study name to evaluate Test Set:"
                    )
                    trial = optimiser.create_fixed_trial(study_name)
                elif choice == 4: 
                    study_name: str = input(
                        "Please enter study to validate:"
                    )
                    optimiser.analyseOptimizerImpact(study_name)
                    optimiser.analyseLearningRate(study_name)
            elif current_menu == "Image Classifier Menu":
                if choice == 0:
                    current_menu = "Main Menu"
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
            elif current_menu == "Fine-Tune Pre-Trained Model Menu":
                if choice == 0:
                    current_menu = "Tweet Classifier Menu"
                elif choice == 1:
                    PretrainedOptimiser(model_name="cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all",
                                        device=Main.DEVICE).run(2)
                elif choice == 2:
                    df: DataFrame = \
                        read_csv(join(STUDIES_DIR, "PretrainedOptimiser_2023-05-23_16-09-28", "results.csv")) \
                        .sort_values(by="value", ascending=False) \
                        .head(1)
                    best_trial_row: Series = df.iloc[0]
                    param_dict: Dict[str, Any] = dict()
                    param_dict.update({
                        "lr": best_trial_row["params_lr"].astype(float),
                        "epochs": best_trial_row["params_epochs"].astype(int),
                        "batch_size": best_trial_row["params_batch_size"].astype(int),
                        "optimiser": best_trial_row["params_optimiser"],
                        "scheduler": best_trial_row["params_scheduler"],
                    })
                    trial: FixedTrial = FixedTrial(param_dict)
                    PretrainedOptimiser(model_name="cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all",
                                        device=Main.DEVICE).test_model(trial)
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")

            print("\n")

    @staticmethod
    def __initial_setup() -> None:
        """
        Performs the initial setup for the application.

        The method checks for the existence of certain directories and creates them if they do not exist.
        The directories checked are those defined in the definitions file.
        """
        if not exists(STUDIES_DIR):
            print(f"Creating directory: {STUDIES_DIR}")
            path: Path = Path(STUDIES_DIR)
            path.mkdir(parents=True, exist_ok=True)
        if not exists(MODELS_DIR):
            print(f"Creating directory: {MODELS_DIR}")
            path: Path = Path(MODELS_DIR)
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def main() -> None:
        """
        The main entry point for the application.

        This method performs the initial setup and then launches the command-line interface.
        """
        # Check if CUDA cache needs to be cleared
        Main.__initial_setup()
        Main.__switch_menu()


if __name__ == "__main__":
    Main.main()

# --------------- Thomas ---------------#
# TODO: Uncomment and integrate with main.py
# Packaged with - dataset.py, resnet.py, resnet_trainer.py
# Author - Thomas Bandy (c3374048)

# from Util.train import Train
# from Util import util
# learn_rates = [0.0001, 0.001, 0.01]
# epochs = [5, 10, 50]
# widths = [10, 50, 100, 500]
# activation_function = ['ReLU', 'Sigmoid', 'CeLU']
# test = Train()

# test.prepare_data()
# util.print_checks(test.train_data, test.valid_data, test.test_data, test.train_loader, test.valid_loader, test.test_loader)
# test.begin_training(10, 2, 0.001)

# --------------- Test all hyper-params ---------------#
# for (w, e, lr) in widths, epochs, learn_rates:
#     test = Train()
#     test.prepare_data()
#     util.print_checks(test.train_data, test.valid_data, test.test_data, test.train_loader, test.valid_loader, test.test_loader)
#     test.begin_training(w, e, lr)

# --------------- Metrics ---------------#
# util.loss_acc_diagram(test.train_losses, test.val_losses, test.train_accs, test.val_accs)
