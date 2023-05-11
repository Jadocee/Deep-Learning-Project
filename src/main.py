from os import system, name
from typing import Dict

from nltk import download
from torch.cuda import is_available as has_cuda

from optimisers.lstm_classifier_optimiser import LSTMClassifierOptimiser


class Main:
    DEVICE: str = "cuda" if has_cuda() else "cpu"

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
        },
        "LSTM Trainer Menu": {
            1: "Optimise Hyperparameters",
        },
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
                    optimiser.run(study_name=study_name if study_name != "" else None)
                else:
                    print("Invalid choice. Try again.")
                    system("cls" if name == "nt" else "clear")
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
        # torch.zeros(1).cuda()


if __name__ == '__main__':
    Main.main()
