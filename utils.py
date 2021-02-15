import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data_adult_census():
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
    return df, X, y


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
    if dataset_name == "adult":
        return get_data_adult_census()
    elif dataset_name == "phishing":
        return get_data_phishing()
    else:
        raise ValueError("unrecognized dataset")


def plt_clear():
    """
    Clears pyplots. Taken from my GATech CS 7646 code.
    """
    plt.clf()
    plt.cla()
    plt.close()


def accuracy_test_size_bar_chart(combined_metrics_dict, metrics_dict_name, test_sizes, title, filename):
    """
    Helper method to make bar charts of accuracy for test sizes.
    Based on: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/custom_ticker1.html#sphx-glr-gallery-ticks-and-spines-custom-ticker1-py
    """
    plt_clear()
    fig, ax = plt.subplots()
    x = np.arange(2)
    y_axis = [combined_metrics_dict[test_sizes[0]][metrics_dict_name]["acc"], combined_metrics_dict[test_sizes[1]][metrics_dict_name]["acc"]]
    plt.bar(test_sizes, y_axis)
    ax.set_xlabel("Test Ratio")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    plt.savefig(filename)
    plt_clear()


def accuracy_two_learners_bar_chart(combined_metrics_dict, test_size, metrics_dict_name_1, metrics_dict_name_2, learner_name_1, learner_name_2, title, filename):
    """
    Helper method to make bar charts of accuracy between two learners.
    """
    plt_clear()
    fig, ax = plt.subplots()
    x = [learner_name_1, learner_name_2]
    y_axis = [combined_metrics_dict[test_size][metrics_dict_name_1]["acc"], combined_metrics_dict[test_size][metrics_dict_name_2]["acc"]]
    plt.bar(x, y_axis)
    ax.set_xlabel("Test Ratio")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    plt.savefig(filename)
    plt_clear()


def accuracy_vs_param_line_chart(x, y, x_label, y_label, title, filename):
    """
    Helper method to make line charts.
    """
    plt_clear()
    fig, ax = plt.subplots()
    plt.plot(x, y, grid=True, marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.savefig(filename)
    plt_clear()
