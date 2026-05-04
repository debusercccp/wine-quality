import pandas as pd

def add_features(df):
    df = df.copy()

    df["high_quality"] = (df["quality"] >= 7).astype(int)
    df = pd.get_dummies(df, columns=["type"], prefix="type")

    return df