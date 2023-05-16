from pandas import read_csv, DataFrame

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
