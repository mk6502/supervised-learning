import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics


logger = logging.getLogger()


def get_data_census():
    """
    Return cleaned adult census data in a dataframe, then also X (attributes) and y (classification).
    """
    class_col = "class"
    na_values = ["?", " ?"]  # "?" is used to denote missing values in this dataset
    df = pd.read_csv("data/adult/adult.csv", na_values=na_values)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["age"] = df["age"].map(lambda x: int(str(x)[0]) if x else None)  # turn age into "decades" - binning
    df = df.dropna()  # ignoring data with blank/null values
    X = df.copy().drop(columns=[class_col])
    y = df.copy()[class_col]
    y = y.apply(lambda x: 1 if x == ">50K" else 0)  # turn y into 0 (<=50K) and 1 (>50K)

    # encode X into float for sklearn:
    encoder_X = OrdinalEncoder()
    encoder_X.fit(X)
    X_encoded = encoder_X.transform(X)

    return df, X_encoded, y


def get_data_phishing():
    """
    Return cleaned phishing data in a dataframe, then also X (attributes) and y (classification).
    """
    class_col = "Result"
    df = pd.read_csv("data/phishing/phishing.csv")  # this data is very clean
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna()  # ignoring data with blank/null values
    X = df.copy().drop(columns=[class_col])
    y = df.copy()[class_col]
    return df, X, y


def get_dataset(dataset_name):
    if dataset_name == "census":
        return get_data_census()
    elif dataset_name == "phishing":
        return get_data_phishing()
    else:
        raise ValueError("unrecognized dataset")


def basic_metrics(y_test, y_pred):
    return {"acc": metrics.accuracy_score(y_test, y_pred)}


def plt_clear():
    """
    Clears plt. Taken from my GATech CS 7646 code.
    """
    plt.clf()
    plt.cla()
    plt.close()


def line_graph(x, y, x_label, y_label, title, filename):
    """
    Helper method to make line charts.
    """
    plt_clear()
    fig, ax = plt.subplots()
    plt.plot(x, y, marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt_clear()


def export_obj_to_json_file(o, filename):
    with open(filename, "w+") as f:
        json.dump(o, f, indent=4, sort_keys=True)
