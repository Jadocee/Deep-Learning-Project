from os import system, name
from typing import Dict


class Main:
    MENUS: Dict[str, Dict[int, str]] = {
        "Main Menu": {
            1: "Image Classifier",
            2: "Tweet Classifier",
        },
        "Tweet Classifier Menu": {
            1: "Train",
            2: "Test",
            3: "Predict",
        }
    }

    def __init__(self) -> None:
        raise Exception("This class is not meant to be instantiated")

    @staticmethod
    def __print_menu(menu: str) -> str:
        if menu not in Main.MENUS.keys():
            raise Exception("Unknown menu")

        for key, value in Main.MENUS[menu].items():
            print(f"{key}. {value}")
        print("0. Main Menu")
        print("-1. Exit")
        choice: int = int(input("Enter choice: "))
        if choice == -1:
            print("Exiting...")
            exit(0)
        elif choice == 0:
            return "Main Menu"
        elif choice not in Main.MENUS[menu].keys():
            print("Invalid choice. Try again.")
            system("cls" if name == "nt" else "clear")
            Main.__print_menu(menu)
        return Main.MENUS[menu][choice]

    @staticmethod
    def main() -> None:
        next_menu: str = Main.__print_menu("Main Menu")
        Main.__print_menu(next_menu)
        # TODO: Implement menu functionality


if __name__ == '__main__':
    Main.main()
