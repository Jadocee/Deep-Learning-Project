from os import system, name, mkdir
from os.path import exists
from typing import Dict

from nltk.downloader import download
from torch.cuda import is_available as has_cuda
from optimisers.bow_classifier_optimiser import BOWClassifierOptimiser

from optimisers.lstm_classifier_optimiser import LSTMClassifierOptimiser
from trainers.bow_classifier_trainer import BOWClassifierTrainer
from utils.definitions import MODELS_DIR, STUDIES_DIR


class Main:
    DEVICE: str = "cuda" if has_cuda() else "cpu"

    MENUS: Dict[str, Dict[int, str]] = {
        "Main Menu": {
            1: "Image Classifier",
            2: "Tweet Classifier",
            3: f"Download NLTK Data",
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
        "BOW Trainer Menu":{
            1: "Run Bag Of Words 1"
        }
    }

    def __init__(self) -> None:
        raise Exception("This class is not meant to be instantiated")

    @staticmethod
    def __display_menu(menu_name: str) -> None:
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
                    optimiser.run(study_name=study_name if study_name != "" else None, prune=True)
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
        # Make sure directories in definitions exist
        if not exists(STUDIES_DIR):
            print(f"Creating directory: {STUDIES_DIR}")
            mkdir(STUDIES_DIR)
        if not exists(MODELS_DIR):
            print(f"Creating directory: {MODELS_DIR}")
            mkdir(MODELS_DIR)

    @staticmethod
    def main() -> None:
        Main.__initial_setup()
        Main.__switch_menu()
        # torch.zeros(1).cuda()


if __name__ == '__main__':
    Main.main()
