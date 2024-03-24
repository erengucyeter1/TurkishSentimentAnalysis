import pandas as pd


def read_lemmantized(file_name):

    data = pd.read_csv("lemmatized.csv")
    data = data.values.tolist()
    return [item for sublist in data for item in sublist]
