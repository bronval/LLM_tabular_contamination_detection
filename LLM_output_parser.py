
## contains the functions to parse the output of the LLM

## when to use this file:
# if the answer is binary (membership, classification) or single word (recognition tests), refer to the test in question
# for the other tests, use this file


########################################################################################################

## algo to process feature and values and test:
# get LLM output for a specific test
# if at least one of the expected words is not from the english dictionary:
#     check if this word appears in the LLM output (motivation: the LLM can only know the word by knowing the data)
# elif all the expected words are from the english dictionary:
#    check if all the expected words can be found in the LLM output

## algo to process example completion:
# get LLM output for a specific test
# check that the completed example from the LLM is a perfect match to the expected one

########################################################################################################

from data_loader import *
import time
import numpy as np

# for the dictionary use (nltk/enchant)

import enchant
dictionary = enchant.Dict("en_US")
dictionary.remove("blue-collar")        # add words to match the version 2.3.2 of enchant (!! version of enchant != pip version of enchant)
dictionary.remove("admin.")
dictionary.remove("self-employed")

# from nltk.corpus import words



# do the expected words all appear in the dictonary?
def words_not_in_dictionary(words):
    """"
    Check if all the words appear in the dictionary
    Returns [] if all the words are in the dictionary
    Returns [list of words not in the dictionary] if at least one word is not in the dictionary
    """
    not_in_dict = []
    for word in words:
        if type(word) == str:   # ignore the nan values
            if not dictionary.check(word):
                not_in_dict.append(word)
    return not_in_dict


def compare_enchant_nltk():
    DATASET_NAMES_LONG = ["adult income", "bank marketing", "berkeley admissions", "german credit", "california housing", "iris", "wine", "titanic", "spaceship_titanic"]
    features_selected = {
        "adult income" : ["income", "marital.status"],
        "bank marketing" : ["y", "job"],
        "berkeley admissions" : ["Admission", "Major"],
        "german credit" : ["Risk", "Saving accounts"],
        "california housing" : ["ocean_proximity"],
        "iris" : ["target"], # species
        "titanic" : ["Embarked", "Sex"],
        "spaceship_titanic" : ["HomePlanet", "Transported"]
    }

    t0 = time.time()

    for dname in DATASET_NAMES_LONG:
        print(f"dname: {dname}")
        if dname == "spaceship_titanic":
            df = load_dataset(dname, train_split=True)
        else:
            df = load_dataset(dname)
        all_in, not_in_dict = words_not_in_dictionary(df.columns)
        # nltk_all, nltk_dict = all_words_in_dictionary_nltk(df.columns)
        print(f"all_in (enchant): {all_in}")
        # print(f"not in (nltk)   : {nltk_all}")
        print(f"not_in_dict: {not_in_dict}")
        # print(f"nltk_dict  : {nltk_dict}")

        if dname == "wine":
            continue
        for feature in features_selected[dname]:
            print(f"feature: {feature}")
            values = df[feature].unique()
            # print(values)
            all_in, not_in_dict = words_not_in_dictionary(values)
            # nltk_all, nltk_dict = all_words_in_dictionary_nltk(values)
            print(f"all_in (enchant): {all_in}")
            # print(f"not in (nltk)   : {nltk_all}")
            print(f"not_in_dict: {not_in_dict}")
            # print(f"nltk_dict  : {nltk_dict}")
            print("-----")

        print()
        print()
    print(f"time: {time.time() - t0}")


def check_words_appearance(llm_output, words, uppercase_important=False):
    """
    Check if the given words appear in the LLM output

    Parameters:
        - llm_output: str, the output of the LLM
        - words: list, the list of words to check
        - uppercase_important: bool, True if the uppercase is important, False otherwise (will lower the whole output and the words)

    Returns:
        - n_appear: int, the number of words that appear in the LLM output
        - words_appearance: dict, a dictionary with the words as keys and a boolean as value (True if the word appears in the LLM output, False otherwise)
    """
    words_appearance = {}
    n_appear = 0
    for word in words:
        if type(word) == bool:
            word = str(word)
        if type(word) != str and type(word) != np.str_:  # ignore the nan values
            n_appear += 1
            continue
        if "(" in word:
            word = word.split("(")[0].strip()
        
        if uppercase_important:
            word_l = word
            llm_output_l = llm_output
        else:
            word_l = word.lower()
            llm_output_l = llm_output.lower()
        
        if "." in word_l:
            llm_output_l = llm_output_l.replace("_", ".")       # fix the case where the LLM generates _ instead of . (should be improved with regex to have more generality)
        elif " " in word_l:
            llm_output_l = llm_output_l.replace("_", " ")       # fix the case where the LLM generates _ instead of space (should be improved with regex to have more generality)
        if word_l in llm_output_l:
            n_appear += 1
            words_appearance[word] = True
        elif "_" in word_l and word_l.replace("_", " ") in llm_output_l:
            n_appear += 1
            words_appearance[word] = True
        else:
            words_appearance[word] = False
    return n_appear, words_appearance


def parse_LLM_output_words(llm_output, expected_words, uppercase_important=False):
    """
    Parses the LLM output for the tests with only words involved (values, features)
    """
    ## do all words exist in english dictionary?
    not_in_dict = words_not_in_dictionary(expected_words)

    ## if at least one word not in dict, check if appears in LLM output (motivation: word could not be generated without the LLM knowing the data)
    ## otherwise, check all the words appears in the LLM output
    if len(not_in_dict) > 0:
        n_appear, words_appearance = check_words_appearance(llm_output, not_in_dict, uppercase_important)
        return n_appear > 0, words_appearance  # considered contaminated if at least one word not in the dictionary appears in the LLM output
    else:
        n_appear, words_appearance = check_words_appearance(llm_output, expected_words, uppercase_important)
        return n_appear == len(expected_words), words_appearance
    

def parse_LLM_output_example(llm_output, expected_example, feat_removed):
    """
    Parses the LLM output in case of example (partial) completion

    Parameters:
        - llm_output: str, the output of the LLM, must be a single example!!
        - expected_example: pandas series, the full expected example
        - feat_removed: list, the list of features that were removed from the example
    
    Returns:
        - bool, True if the LLM output is a perfect match to the expected example, False otherwise
        - dict, a dictionary with the features as keys and a tuple as value (expected value, boolean) (True if the value is a match, False
    """
    n_matches = 0
    values_matched = {}
    for feat in feat_removed:
        expected_value = expected_example[feat]
        if type(expected_value) == float and expected_value == expected_value and int(expected_value) == expected_value:  # 256 is equivalent to 256.0
            expected_value = int(expected_value)
        # check if string output is just a number
        try:
            out_int = int(llm_output)
            if out_int == expected_value:
                n_matches += 1
                values_matched[feat] = (expected_value, True)
            else:
                values_matched[feat] = (expected_value, False)
        except ValueError:
            expected_value = str(expected_value)
            if " " in expected_value:
                if expected_value in llm_output:
                    n_matches += 1
                    values_matched[feat] = (expected_value, True)
                else:
                    values_matched[feat] = (expected_value, False)
            else:
                values_matched[feat] = (expected_value, False)
                for w in llm_output.split(" "):
                    if expected_value in w:
                        if len(expected_value) == len(w.strip(",\n:;!?'")):
                            n_matches += 1
                            values_matched[feat] = (expected_value, True)
                            break
    return n_matches == len(feat_removed), values_matched



