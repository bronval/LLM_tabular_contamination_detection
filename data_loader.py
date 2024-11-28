
### Contains the functions to load the different datasets ###

import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.utils import shuffle


def load_dataset(dname : str, train_split : bool = False) -> pd.DataFrame:
    if dname == "berkeley" or dname == "berkeley admissions":
        return pd.read_csv("datasets/berkeley.csv")
    elif dname == "german credit" or dname == "german":
        df = pd.read_csv("datasets/german.csv")
        df.drop(columns=["Unnamed: 0"], inplace=True)
        return df
    elif dname == "bank marketing" or dname == "bank":
        return pd.read_csv("datasets/bank.csv")
    elif dname == "adult" or dname == "adult income":
        # return pd.read_csv("datasets/adult.csv")
        return pd.read_csv("datasets/adult_v2.csv")
    elif dname == "california housing" or dname == "housing":
        return pd.read_csv("datasets/housing.csv")
    elif dname == "iris":
        data = load_iris(as_frame=True)
        target_names = data.target_names
        df = data.frame
        df["target"] = df["target"].apply(lambda x: target_names[x])
        return shuffle(df, random_state=0)
    elif dname == "wine":
        data = load_wine(as_frame=True)
        target_names = data.target_names
        df = data.frame
        df["target"] = df["target"].apply(lambda x: target_names[x])
        return shuffle(df, random_state=0)
    elif dname == "titanic":
        # if train_split:
        return pd.read_csv("datasets/titanic/titanic_train.csv")    # use train split because not all features are available in test split (note: results similar on test split)
        # else:
            # return pd.read_csv("datasets/titanic/titanic_test.csv")
    elif dname == "spaceship_titanic":
        # if train_split:
        return pd.read_csv("datasets/spaceship_titanic/spaceship_titanic_train.csv")    # use train split because not all features are available in test split (note: results similar on test split)
        # else:
            # return pd.read_csv("datasets/spaceship_titanic/spaceship_titanic_test.csv")
    elif dname == "MathE":
        df = pd.read_csv("datasets/MathE.csv", sep=";")
        df = df.drop_duplicates(subset=df.columns[1:])  # drop the duplicates without looking at the student id
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the dataset to avoid selection of the same values
        return df
    elif dname == "thyroid_diff" or dname == "thyroid disease recurrence":
        df = pd.read_csv("datasets/Thyroid_Diff.csv")
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        return df
    else:
        raise ValueError(f"Unknown dataset: {dname}")
    


