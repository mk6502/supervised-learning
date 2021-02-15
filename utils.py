import pandas as pd
import matplotlib.pyplot as plt


def get_data_adult_census():
    """
    Return cleaned adult census data in a dataframe, then also X (attributes) and y (classification).
    """
    na_values = ["?", " ?"]  # "?" is used to denote missing values in this dataset
    df = pd.read_csv("data/adult/adult.csv", na_values=na_values)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["age"] = df["age"].map(lambda x: int(str(x)[0]) if x else None)  # turn age into "decades" - binning
    df = df.dropna()  # ignoring data with blank/null values
    X = df.copy().drop(columns=["class"])
    y = df.copy()["class"]
    y = y.apply(lambda x: 1 if x == ">50K" else 0)  # turn y into 0 (<=50K) and 1 (>50K)
    return df, X, y


def get_data_some_other():
    """
    Return cleaned data for another dataset in a dataframe, then also X (attributes) and y (classification).
    """
    # return df, X, y
    return None  # TODO!


def get_dataset(dataset_name):
    if dataset_name == "adult":
        return get_data_adult_census()
    else:
        return get_data_some_other()


def plt_clear():
    """
    Taken from my GATech CS7646 code.
    """
    plt.clf()
    plt.cla()
    plt.close()
