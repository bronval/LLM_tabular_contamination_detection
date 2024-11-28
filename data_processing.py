
### Contains the functions to process data, before and after the LLM ###

import pandas as pd
import numpy as np
from data_loader import load_dataset
# from prompts_no_context import *
from prompts import *
import random


TARGETS_FEATURES = {
    "adult income": "income",
    "bank marketing": "y",
    "berkeley admissions": "Admission",
    "german credit": "Risk",
    "california housing": "median_house_value",
    "iris": "target",
    "wine": "target",
    "titanic": "Survived",
    "spaceship_titanic": "Transported",
    "thyroid disease recurrence": "Recurred",
    "MathE": None,                  # not a classification task
}



def serialize_sample(sample : pd.Series,
                     columns_ignored : list = [],
                     column_names : dict = {},
                     features_sep : str = '. ',
                     value_sep : str = ' is ',
                     specifier : str = 'The ',
                     remove_last_sep : bool = False) -> str:
    """
    Create a text from the sample given in input.
    Serialization follows the format: "[feature name][value_sep][feature value][features_sep] " if column_names is empty.
    Otherwise, the column_names will be used instead of the feature names to allow clearer feature names.
    Warning: no whitespace are included between feature-sep-value and should be given in the parameters.

    Args:
        - sample: pandas Series representing the sample to be serialized
        - columns_ignored: list of column names to be ignored during serialization (default: []). Names must be the original ones, not the ones given in column_names
        - column_names: a mapping between the feature names in df and the custom feature names to use (default: {}). If empty, the feature names from the sample will be used.
        - features_sep: separator between each feature-value couple (default: '.')
        - value_sep: separator between feature and value (default: ' is ')
        - specifier: to be put before the feature name (default: '')
        - remove_last_sep: whether to remove the last features_sep (default: False)

    Returns:
        - a string representing the serialized sample
    """
    sample = sample.drop(columns_ignored, axis=0)

    if len(column_names) == 0:
        serialized_sample =  "".join([f"{specifier}{feat}{value_sep}{value}{features_sep}" for feat, value in sample.items()])
        if remove_last_sep:
            serialized_sample = serialized_sample[:-len(features_sep)]

    else:
        if len(column_names) - len(columns_ignored) != len(sample):
            raise ValueError(f"column_names and sample must have the same length, {len(column_names) - len(columns_ignored)} != {len(sample)}")
        serialized_sample = "".join([f"{specifier}{column_names[feat]}{value_sep}{value}{features_sep}" for feat, value in sample.items()])
        if remove_last_sep:
            serialized_sample = serialized_sample[:-len(features_sep)]
    return serialized_sample


def serialize_samples(df : pd.DataFrame,
                      n_samples : int = 5,
                      random_sampling : bool = False,
                      seed : int = 0,
                      column_names : dict = {},
                      features_sep : str = ", ",
                      value_sep : str = " is ",
                      specifier : str = "The ",
                      remove_last_sep : bool = False) -> str:
    """
    Serializes n_samples from the DataFrame df.
    Parameters:
        - df: DataFrame with all the data
        - n_samples: number of samples to be serialized
        - random_sampling: whether to sample the data randomly or not
        - seed: seed for the random sampling
        - column_names: a mapping between the feature names in df and the custom feature names to use (default: {}). If empty, the feature names from the sample will be used.
        - features_sep: separator between each feature-value couple (default: ', ')
        - value_sep: separator between feature and value (default: ' is ')
        - specifier: to be put before the feature name (default: 'The ')
        - remove_last_sep: whether to remove the last features_sep (default: False)
    
    Returns a simple string with all the serialized examples
    """

    result = ""
    if random_sampling:
        df = df.sample(n=n_samples, random_state=seed)
    else:
        df = df.iloc[:n_samples]
    for i in range(n_samples):
        result += serialize_sample(df.iloc[i],
                                       column_names=column_names,
                                       features_sep=features_sep,
                                       value_sep=value_sep,
                                       specifier=specifier,
                                       remove_last_sep=remove_last_sep)
        result += "\n"
    return result.rstrip("\n")

                           
def get_prompt(prompt : str, infos : list) -> str:
    """
    infos is a list of strings to be inserted in the prompt.
    It should contain the information in the following order:
    0) dataset name
    1) feature name(s)
    2) serialized example(s)
    """
    if len(infos) == 0:
        raise ValueError("infos must contain at least one element")
    elif len(infos) == 1:
        return prompt.format(dname=infos[0])
    elif len(infos) == 2:
        if infos[0] != None:
            return prompt.format(dname=infos[0], fname=infos[1])
        else:
            return prompt.format(fname=infos[1])
    elif len(infos) == 3:
        if infos[0] is None and infos[1] is None:
            return prompt.format(serialization=infos[2])
        elif infos[1] is None:
            return prompt.format(dname=infos[0], serialization=infos[2])
        return prompt.format(dname=infos[0], fname=infos[1], serialization=infos[2])
    elif len(infos) == 4:
        if infos[2] is None:
            return prompt.format(dname=infos[0], fname=infos[1], values=infos[3])
        elif infos[1] is None:
            return prompt.format(dname=infos[0], serialization=infos[2], example=infos[3])
        else:
            return prompt.format(dname=infos[0], fname=infos[1], serialization=infos[2], example=infos[3])
    else:
        raise ValueError("infos must contain at most three elements")


def convert_target(dname, target):
    """
    Given the dataset name and the target value, returns the corresponding answer for the classification with a yes or no question.

    Parameters:
        - dname: name of the dataset
        - target: target value to be converted
    
    Returns a string with the expected answer.
    """
    if dname == "adult income":
        return "Yes" if target == ">50K" else "No"
    elif dname == "bank marketing":
        return "Yes" if target == "yes" else "No"
    elif dname == "berkeley admissions":
        return "Yes" if target == "Accepted" else "No"
    elif dname == "german credit":
        return "Yes" if target == "good" else "No"
    elif dname == "california housing":
        return "Yes" if target >= 179700.0 else "No"
    elif dname == "iris":
        return target
    elif dname == "wine":
        return str(target)[len("class_"):]
    elif dname == "titanic":
        return "Yes" if target == 1 else "No"
    elif dname == "spaceship_titanic":
        return "Yes" if target == True else "No"
    elif dname == "thyroid_diff" or dname == "thyroid disease recurrence":
        return target
    else:
        raise ValueError(f"Dataset {dname} not recognized")


def get_prompt_classif(dname, serialization):
    """
    Gives the prompt for the classification task for the given dataset.

    Parameters:
        - dname: name of the dataset
        - serialization: serialized examples to be inserted in the prompt
    
    Returns the prompt as a string.
    """
    df_prompts = {
        "adult income": prompt_classif_adult,
        "bank marketing": prompt_classif_bank,
        "berkeley admissions": prompt_classif_berkeley,
        "german credit": prompt_classif_german,
        "california housing": prompt_classif_housing,
        "iris": prompt_classif_iris,
        "wine": prompt_classif_wine,
        "titanic": prompt_classif_titanic,
        "spaceship_titanic": prompt_classif_sptitanic,
        "thyroid disease recurrence": prompt_classif_thyroid,
    }
    prompt = df_prompts[dname]
    return prompt.format(serialization=serialization)


def get_prompt_classif_shots(dname, serialization):
    """
    Gives the prompt for the classification task for the given dataset with examples with few few shots

    Parameters:
        - dname: name of the dataset
        - serialization: serialized examples to be inserted in the prompt
    
    Returns the prompt as a string.
    """
    df_prompts_shots= {
        "adult income": prompt_classif_adult_shots,
        "bank marketing": prompt_classif_bank_shots,
        "berkeley admissions": prompt_classif_berkeley_shots,
        "german credit": prompt_classif_german_shots,
        "california housing": prompt_classif_housing_shots,
        "iris": prompt_classif_iris_shots,
        "wine": prompt_classif_wine_shots,
        "titanic": prompt_classif_titanic_shots,
        "spaceship_titanic": prompt_classif_sptitanic_shots,
        "thyroid disease recurrence": prompt_classif_thyroid_shots,
    }
    df_sentences = {
        "adult income": "Does this person earn more than 50K per year? Yes or no?",
        "bank marketing": "Does this client subscribe to a term deposit? Yes or no?",
        "berkeley admissions": "Is this student accepted to the university? Yes or no?",
        "german credit": "Is this client a good credit risk? Yes or no?",
        "california housing": "Is this house expensive? Yes or no?",
        "iris": "What is the species of this iris? Setosa, Versicolor or Virginica?",
        "wine": "What is the quality of this wine? 0, 1 or 2?",
        "titanic": "Did this passenger survive the Titanic disaster? Yes or no?",
        "spaceship_titanic": "Was this passenger transported to the destination? Yes or no?",
        "thyroid disease recurrence": "Did this patient have a thyroid disease recurrence? Yes or no?",
    }
    df = load_dataset(dname)
    target = TARGETS_FEATURES[dname]
    values = df[target].unique()
    if dname == "california housing":
        values = ["Yes", "No"]
    example = ""
    for val in values:
        if dname == "california housing":
            if val == "Yes":
                sample = df[df[target] >= 179700.0].iloc[0]
            else:
                sample = df[df[target] < 179700.0].iloc[0]
        else:
            sample = df[df[target] == val].iloc[0]
        y = sample[target]
        sample = sample.drop(target)
        example += "Example:\n" + serialize_sample(sample) + "\n" + df_sentences[dname] + "\nAnswer: " + convert_target(dname, y) + "\n\n"
    return df_prompts_shots[dname].format(serialization=serialization, example=example)


def evaluate_output_classif(dname, out, y):
    """
    Evaluate the output of the LLM for the classification task.

    Parameters:
        - dname: name of the dataset
        - out: output of the LLM
        - y: target value

    Returns True if the output is correct, False otherwise. 
    """
    try:
        if out == "":
            print(f"empty output ({dname}, {y})")
            return False
        if dname != "iris":
            out = out[:3]
        if dname == "adult income":
            if ("yes" in out.lower() or "1" in out.lower()) and y == ">50K":
                return True
            elif ("no" in out.lower() or "0" in out.lower()) and y == "<=50K":
                return True
            else:
                return False
        elif dname == "bank marketing":
            if ("yes" in out.lower() or "1" in out.lower()) and y == "yes":
                return True
            elif ("no" in out.lower() or "0" in out.lower()) and y == "no":
                return True
            else:
                return False
        elif dname == "berkeley admissions":
            if ("yes" in out.lower() or "1" in out.lower()) and y == "Accepted":
                return True
            elif ("no" in out.lower() or "0" in out.lower()) and y == "Rejected":
                return True
            else:
                return False
        elif dname == "german credit":
            if ("yes" in out.lower() or "1" in out.lower()) and y == "good":
                return True
            elif ("no" in out.lower() or "0" in out.lower()) and y == "bad":
                return True
            else:
                return False
        elif dname == "california housing":
            median_value = 179700.0
            if ("yes" in out.lower() or "1" in out.lower()) and y >= median_value:
                return True 
            elif ("no" in out.lower() or "0" in out.lower()) and y < median_value:
                return True
            else:
                return False
        elif dname == "iris":
            if y.lower() in out.lower():
                return True
            else:
                return False
        elif dname == "wine":
            answer = "class_" + out.lower()
            if str(y) in answer:
                return True
            else:
                return False
        elif dname == "titanic":
            if ("yes" in out.lower() or "1" in out.lower()) and y == 1:
                return True
            elif ("no" in out.lower() or "0" in out.lower()) and y == 0:
                return True
            else:
                return False
        elif dname == "spaceship_titanic":
            if ("yes" in out.lower() or "1" in out.lower()) and y == True:
                return True
            elif ("no" in out.lower() or "0" in out.lower()) and y == False:
                return True
            else:
                return False
        elif dname == "thyroid_diff" or dname == "thyroid disease recurrence":
            if ("yes" in out.lower() or "1" in out.lower()) and y == "Yes":
                return True
            elif ("no" in out.lower() or "0" in out.lower()) and y == "No":
                return True
            else:
                return False
        else:
            raise ValueError(f"Dataset {dname} not recognized")
    except Exception as e:
        print(f"Error while parsing answer for {dname}: {e}\n(out: {out}, y: {y})")
        return False


def shuffle_sample(sample : pd.DataFrame, columns, n_swapped=2):
    """
    Given a sample from a dataset, swaps n_swapped values between two columns without swapping the feature names.

    Parameters:
        - sample: pandas DataFrame representing the sample
        - columns: list of columns to be shuffled
        - n_swapped: number of values to be swapped (default: 2)

    returns the shuffled sample as a pandas DataFrame
    """

    out = sample.copy()
    swapped = set()
    for i in range(n_swapped):
        # select 2 columns for which values will be swapped
        col1 = np.random.choice(columns)
        col2 = np.random.choice(columns)
        while col1 == col2 or (col1, col2) in swapped or (col2, col1) in swapped:
            col2 = np.random.choice(columns)
        swapped.add((col1, col2))
        # swap values
        out[col1], out[col2] = out[col2], out[col1]
    return out


def shuffle_example_value(df : pd.DataFrame, sample : pd.DataFrame, n_swapped=5):

    # select n_swapped columns for which values will be swapped with other value from the same column
    possible_cols = []
    for col in df.columns:
        if len(df[col].unique()) > 1:
            possible_cols.append(col)
    
    # check if a change of the n_swapped value is needed
    if n_swapped > len(possible_cols):
        print(F"WARNING: n_swapped {n_swapped} is greater than the number of columns {len(df.columns)}. Set to len(df.columns)-2 {len(df.columns)-2}")
        n_swapped = len(possible_cols) - 2
        if n_swapped < 1:
            print("WARNING: n_swapped is less than 1. Set to 1")
            n_swapped = 1
    columns = random.sample(possible_cols, n_swapped)

    # create a copy of the sample to modify it so that it is not in the dataset
    res = sample.copy()

    # change the example while it matches an example in the dataset
    while (res == df).all(1).any():
        for col in columns:
            actual_value = res[col]

            # obtain a new value and make sure it is different from the actual value in the example
            new_value = random.choice(df[col].values)
            while new_value == actual_value:
                new_value = random.choice(df[col].values)
            
            # set the new value for our swapped example
            res[col] = new_value

    return res


def convert_results_csv(dict_results):
    """
    Converts the results from the dictionary format to a pandas DataFrame.
    """
    # df = pd.DataFrame(columns=["test_name", "Adult", "Bank", "Berkeley", "German", "Housing", "Iris", "Wine", "Titanic", "Sp. Titanic", "Thyroid", "MathE"])
    df = pd.DataFrame(columns=["test_name", "Thyroid_Diff", "MathE"])
    
    dname_matching = {
        "adult income": "Adult",
        "bank marketing": "Bank",
        "berkeley admissions": "Berkeley",
        "german credit": "German",
        "california housing": "Housing",
        "iris": "Iris",
        "wine": "Wine",
        "titanic": "Titanic",
        "spaceship_titanic": "Sp. Titanic",
        "thyroid disease recurrence": "Thyroid_Diff",
        "MathE": "MathE"
    }
    for test_name, results in dict_results.items():
        row = {"test_name": test_name}
        for dname, val in results.items():
            row[dname_matching[dname]] = int(val)
        df = df._append(row, ignore_index=True)
    df.fillna(0, inplace=True)
    return df





