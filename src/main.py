from os import system, name
from typing import Dict

from nltk import download
from torch.cuda import is_available as has_cuda

from optimisers.lstm_classifier_optimiser import LSTMClassifierOptimiser
from trainers.bow_classifier_trainer import BOWClassifierTrainer


class Main:
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
    def switch_menu():
        current_menu = "Main Menu"

        while True:
            Main.__display_menu(current_menu)
            choice: int = int(input("Enter choice: "))
            print("\n")

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
                    optimiser: LSTMClassifierOptimiser = LSTMClassifierOptimiser()
                    optimiser.run()
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
            elif current_menu == "BOW Trainer Menu":
                if choice == 0:
                        current_menu = "Main Menu"
                elif choice == 1:
                    print({"BagOfWords"})
                    classifier: BOWClassifierTrainer = BOWClassifierTrainer()
                    classifier.run()
            elif current_menu == "Image Classifier Menu":
                if choice == 0:
                    current_menu = "Main Menu"
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")

            print("\n")

    @staticmethod
    def main() -> None:
        Main.switch_menu()
        # TODO: Implement menu functionality
        # TODO: Add option to download nltk data (stopwords, etc.)
        # Temporary solution: uncomment the following line to download missing NLTK stopwords
        # nltk.download("stopwords")
        # nltk.download("punkt")


if __name__ == '__main__':
    Main.main()
