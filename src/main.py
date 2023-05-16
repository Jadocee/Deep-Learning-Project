from pathlib import Path

from nltk.downloader import download
from os import system, name, mkdir
from os.path import exists
from torch.cuda import is_available as has_cuda
from typing import Dict

from optimisers.bow_classifier_optimiser import BOWClassifierOptimiser
from optimisers.lstm_classifier_optimiser import LSTMClassifierOptimiser
from utils.definitions import MODELS_DIR, STUDIES_DIR


class Main:
    """
    Main class which acts as the command-line interface of the application.

    This class is not meant to be instantiated. It provides several static methods for interacting with the application.
    """

    DEVICE: str = "cuda" if has_cuda() else "cpu"
    """
    The device to be used for computations. If CUDA is available, it defaults to "cuda", else "cpu".
    """

    MENUS: Dict[str, Dict[int, str]] = {
        "Main Menu": {
            1: "Image Classifier",
            2: "Tweet Classifier",
            3: "Download NLTK Data",
            4: "Check CUDA Availability",
        },
        "Image Classifier Menu": {
            1: "TODO",
        },
        "Tweet Classifier Menu": {
            1: "LSTM Trainer",
            2: "BOW Trainer"
        },
        "LSTM Trainer Menu": {
            1: "Optimise Hyperparameters",
        },
        "BOW Trainer Menu": {
            1: "Run Bag Of Words 1"
        }
    }
    """
    A dictionary containing the structure of the menus to be displayed on the command-line interface.
    """

    def __init__(self) -> None:
        """
        The constructor for Main class. Raises an exception since the class should not be instantiated.

        Raises:
            Exception: If an attempt is made to instantiate the Main class.
        """
        raise Exception("This class is not meant to be instantiated")

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

        menu: Dict[int, str] = Main.MENUS[menu_name]
        print(f"--- {menu_name} ---")
        for key, value in menu.items():
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
            choice: int = int(input("Enter choice: "))

            if choice == -1:
                print("Exiting...")
                break

            if current_menu == "Main Menu":
                if choice == 1:
                    current_menu = "Image Classifier Menu"
                elif choice == 2:
                    current_menu = "Tweet Classifier Menu"
                elif choice == 3:
                    download("stopwords")
                    download("punkt")
                    download("wordnet")
                elif choice == 4:
                    print(f"CUDA is {'available' if has_cuda() else 'not available'}")
                    input("Press any key to continue...")
                    system("cls" if name == "nt" else "clear")
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
            elif current_menu == "Tweet Classifier Menu":
                if choice == 0:
                    current_menu = "Main Menu"
                elif choice == 1:
                    current_menu = "LSTM Trainer Menu"
                elif choice == 2:
                    current_menu = "BOW Trainer Menu"
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
            elif current_menu == "LSTM Trainer Menu":
                if choice == 0:
                    current_menu = "Main Menu"
                elif choice == 1:
                    system("cls" if name == "nt" else "clear")
                    study_name: str = input("Enter study name or press enter to use default: ")
                    optimiser: LSTMClassifierOptimiser = LSTMClassifierOptimiser(device=Main.DEVICE)
                    optimiser.run(study_name=study_name if study_name != "" else None, prune=True, n_trials=150)
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
            elif current_menu == "BOW Trainer Menu":
                if choice == 0:
                    current_menu = "Main Menu"
                elif choice == 1:
                    print({"BagOfWords"})
                    optimiser: BOWClassifierOptimiser = BOWClassifierOptimiser(device=Main.DEVICE)
                    optimiser.run(None, prune=True)
            elif current_menu == "Image Classifier Menu":
                if choice == 0:
                    current_menu = "Main Menu"
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
