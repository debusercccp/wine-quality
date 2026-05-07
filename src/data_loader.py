import pandas as pd

def load_data():
    red = pd.read_csv("data/winequality-red.csv", sep=";")
    white = pd.read_csv("data/winequality-white.csv", sep=";")

    red["type"] = "red"
    white["type"] = "white"

    df = pd.concat([red, white], ignore_index=True)

    return df
